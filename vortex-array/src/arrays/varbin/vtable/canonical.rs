// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use crate::arrays::varbinview::build_views::build_views_from_offsets;
use crate::match_each_integer_ptype;

/// Converts a VarBinArray to its canonical form (VarBinViewArray).
///
/// This is a shared helper used by both `canonicalize` and `execute`.
///
/// The value bytes are handed over as they are — only the offsets are consumed, to derive the view
/// lengths — so this costs one view per row and no byte copy.
pub(crate) fn varbin_to_canonical(
    array: ArrayView<'_, VarBin>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<VarBinViewArray> {
    let parts = array.into_owned().into_data_parts();
    let offsets = parts.offsets.execute::<PrimitiveArray>(ctx)?;
    let (buffers, views) = match_each_integer_ptype!(offsets.ptype(), |P| {
        build_views_from_offsets(
            0,
            MAX_BUFFER_LEN,
            parts.bytes.unwrap_host(),
            offsets.as_slice::<P>(),
        )
    });

    // SAFETY: views are correctly computed from valid offsets
    Ok(unsafe {
        VarBinViewArray::new_unchecked(views, Arc::from(buffers), parts.dtype, parts.validity)
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ChunkedArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::varbin::builder::VarBinBuilder;
    use crate::assert_arrays_eq;
    use crate::builders::VarBinViewBuilder;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    #[rstest]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Binary(Nullability::Nullable))]
    fn test_canonical_varbin_sliced(#[case] dtype: DType) {
        let mut varbin = VarBinBuilder::<i32>::with_capacity(dtype.clone(), 10);
        varbin.push_null();
        varbin.push_null();
        // inlined value
        varbin.append_value("123456789012".as_bytes());
        // non-inlinable value
        varbin.append_value("1234567890123".as_bytes());
        let varbin = varbin.finish_into_varbin();

        let varbin = varbin.slice(1..4).unwrap();

        let mut ctx = array_session().create_execution_ctx();
        let canonical = varbin.execute::<VarBinViewArray>(&mut ctx).unwrap();
        assert_eq!(canonical.dtype(), &dtype);

        assert!(
            !canonical
                .is_valid(0, &mut array_session().create_execution_ctx())
                .unwrap()
        );

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
        let mut ctx = array_session().create_execution_ctx();
        let varbin = VarBinArray::from_iter_nonnull(["foo", "bar", "baz"], dtype.clone());
        let canonical = varbin
            .as_array()
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)
            .unwrap();
        let expected = match dtype {
            DType::Utf8(_) => VarBinViewArray::from_iter_str(["foo", "bar", "baz"]),
            _ => VarBinViewArray::from_iter_bin(["foo", "bar", "baz"]),
        };
        assert_arrays_eq!(canonical, expected, &mut ctx);
    }

    /// Appending a `VarBin` array to a `VarBinViewBuilder` builds views over its bytes directly
    /// instead of canonicalizing first, so the views must be numbered against the buffers the
    /// builder already holds. Interleaving `VarBin` appends with value appends (which stage an
    /// in-progress buffer) and with a `VarBinView` append exercises that numbering.
    #[rstest]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Binary(Nullability::Nullable))]
    fn append_varbin_to_varbinview_builder(#[case] dtype: DType) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let long = "a value long enough that its view has to reference a buffer";
        let longer = "another value long enough that its view has to reference a buffer";

        // Two chunks, each with an inlined value, a buffer-referencing value and a null, so both
        // pushed buffers are non-empty and the second must not reuse the first's index.
        let first = VarBinArray::from_iter([Some("short"), None, Some(long)], dtype.clone());
        let second = VarBinArray::from_iter([Some(longer), Some("tiny"), None], dtype.clone());
        let view = VarBinViewArray::from_iter([Some(long), None], dtype.clone());

        let mut builder = VarBinViewBuilder::with_capacity(dtype.clone(), 8);
        first
            .as_array()
            .clone()
            .append_to_builder(&mut builder, &mut ctx)?;
        // Stages an in-progress buffer, which the next append has to account for.
        builder.append_value(longer);
        second
            .as_array()
            .clone()
            .append_to_builder(&mut builder, &mut ctx)?;
        view.clone()
            .into_array()
            .append_to_builder(&mut builder, &mut ctx)?;

        let expected = ChunkedArray::try_new(
            vec![
                first.as_array().clone(),
                VarBinViewArray::from_iter([Some(longer)], dtype.clone()).into_array(),
                second.as_array().clone(),
                view.into_array(),
            ],
            dtype,
        )?;
        assert_arrays_eq!(builder.finish_into_varbinview(), expected, &mut ctx);
        Ok(())
    }

    /// A builder configured to compact must not be handed a raw buffer behind its back: the
    /// inlined values leave the pushed buffer only partly referenced, and skipping compaction
    /// would keep those bytes alive. The builder measures utilization from the value lengths and
    /// drops the fully-inlined heap.
    #[test]
    fn append_varbin_to_a_compacting_builder_still_compacts() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Utf8(Nullability::NonNullable);
        // Every value inlines, so nothing references the value bytes at all.
        let array = VarBinArray::from_iter_nonnull(["short", "tiny", "small"], dtype.clone());

        let mut builder = VarBinViewBuilder::with_compaction(dtype, 4, 1.0);
        array
            .as_array()
            .clone()
            .append_to_builder(&mut builder, &mut ctx)?;
        let compacted = builder.finish_into_varbinview();

        assert!(
            compacted
                .data_buffers()
                .iter()
                .all(|buffer| buffer.is_empty()),
            "a fully-inlined append should not retain any value bytes"
        );
        assert_arrays_eq!(compacted, array.as_array().clone(), &mut ctx);
        Ok(())
    }

    // Empty array: offsets has exactly one element; no elements to canonicalize.
    #[test]
    fn test_canonical_varbin_empty() {
        let varbin =
            VarBinArray::from_iter_nonnull([] as [&str; 0], DType::Utf8(Nullability::NonNullable));
        let mut ctx = array_session().create_execution_ctx();
        let canonical = varbin
            .as_array()
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)
            .unwrap();
        assert_eq!(canonical.len(), 0);
    }
}

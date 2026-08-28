// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::VariantArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::variant::VariantArraySlotsExt;
use crate::builders::builder_with_capacity_in;
use crate::dtype::DType;

pub(super) fn _canonicalize(
    array: ArrayView<'_, Chunked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Canonical> {
    if array.nchunks() == 0 {
        if matches!(array.dtype(), DType::Variant(_)) {
            return VariantArray::try_new(array.array().clone().into_array(), None)
                .map(Canonical::Variant);
        }
        return Ok(Canonical::empty(array.dtype()));
    }
    if array.nchunks() == 1 {
        return array.chunk(0).clone().execute::<Canonical>(ctx);
    }

    Ok(match array.dtype() {
        DType::Variant(_) => {
            let owned_chunks: Vec<ArrayRef> = array.iter_chunks().cloned().collect();
            Canonical::Variant(pack_variant_chunks(owned_chunks, ctx)?)
        }
        _ => {
            let mut builder = builder_with_capacity_in(ctx.allocator(), array.dtype(), array.len());
            array.array().append_to_builder(builder.as_mut(), ctx)?;
            builder.finish_into_canonical(ctx)
        }
    })
}

/// Packs many [`VariantArray`]s into one [`VariantArray`] with chunked children.
///
/// The caller guarantees there are at least 2 chunks.
fn pack_variant_chunks(
    chunks: Vec<ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<VariantArray> {
    let variant_chunks: Vec<VariantArray> = chunks
        .into_iter()
        .map(|chunk| chunk.execute::<VariantArray>(ctx))
        .try_collect()?;

    let outer_dtype = variant_chunks[0].dtype().clone();
    let core_storage = ChunkedArray::try_new(
        variant_chunks
            .iter()
            .map(|chunk| chunk.core_storage().clone()),
        outer_dtype,
    )?
    .into_array();

    let shredded = match variant_chunks[0].shredded() {
        None => {
            for chunk in &variant_chunks[1..] {
                vortex_ensure!(
                    chunk.shredded().is_none(),
                    "cannot canonicalize ChunkedArray<Variant>: chunks disagree on shredded presence"
                );
            }
            None
        }
        Some(first_shredded) => {
            let shredded_dtype = first_shredded.dtype().clone();
            let mut shredded_chunks = Vec::with_capacity(variant_chunks.len());
            shredded_chunks.push(first_shredded.clone());

            for chunk in &variant_chunks[1..] {
                let shredded = chunk.shredded().ok_or_else(|| {
                    vortex_err!(
                        "cannot canonicalize ChunkedArray<Variant>: chunks disagree on shredded presence"
                    )
                })?;
                vortex_ensure!(
                    shredded.dtype() == &shredded_dtype,
                    "cannot canonicalize ChunkedArray<Variant>: shredded dtype mismatch ({} vs {})",
                    shredded_dtype,
                    shredded.dtype()
                );
                shredded_chunks.push(shredded.clone());
            }

            Some(ChunkedArray::try_new(shredded_chunks, shredded_dtype)?.into_array())
        }
    };

    VariantArray::try_new(core_storage, shredded)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Chunked;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::FixedSizeList;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListView;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::Struct;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::VariantArray;
    use crate::arrays::chunked::ChunkedArrayExt;
    use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
    use crate::arrays::listview::ListViewArraySlotsExt;
    use crate::arrays::struct_::StructArrayExt;
    use crate::arrays::variant::VariantArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType::List;
    use crate::dtype::DType::Primitive;
    use crate::dtype::DType::Variant as VariantDType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType::I32;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    /// A shared session for these chunked-array tests, used to create execution contexts.
    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

    fn variant_scalar(value: i32) -> Scalar {
        Scalar::variant(Scalar::primitive(value, NonNullable))
    }

    fn variant_core(values: impl IntoIterator<Item = i32>) -> VortexResult<ArrayRef> {
        Ok(ChunkedArray::try_new(
            values
                .into_iter()
                .map(|value| ConstantArray::new(variant_scalar(value), 1).into_array()),
            VariantDType(NonNullable),
        )?
        .into_array())
    }

    fn variant_chunk(values: impl IntoIterator<Item = i32>) -> VortexResult<VariantArray> {
        VariantArray::try_new(variant_core(values)?, None)
    }

    fn variant_chunk_with_shredded(
        values: impl IntoIterator<Item = i32>,
        shredded: ArrayRef,
    ) -> VortexResult<VariantArray> {
        VariantArray::try_new(variant_core(values)?, Some(shredded))
    }

    fn into_variant(canonical: Canonical) -> VortexResult<VariantArray> {
        match canonical {
            Canonical::Variant(array) => Ok(array),
            other => vortex_bail!("expected Variant canonical array, got {other:?}"),
        }
    }

    fn assert_variant_values(array: &VariantArray, expected: &[i32]) -> VortexResult<()> {
        assert_eq!(array.len(), expected.len());
        let mut ctx = SESSION.create_execution_ctx();

        for (idx, expected) in expected.iter().copied().enumerate() {
            let scalar = array.execute_scalar(idx, &mut ctx)?;
            let actual = scalar
                .as_variant()
                .value()
                .and_then(|value| value.as_primitive().as_::<i32>());
            assert_eq!(actual, Some(expected), "row {idx}");
        }

        Ok(())
    }

    #[test]
    fn pack_variant_chunks_without_shredded() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let chunked = ChunkedArray::try_new(
            vec![
                variant_chunk([1, 2])?.into_array(),
                variant_chunk([3])?.into_array(),
            ],
            VariantDType(NonNullable),
        )?
        .into_array();

        let variant = into_variant(chunked.execute::<Canonical>(&mut ctx)?)?;

        assert_eq!(variant.len(), 3);
        assert!(variant.shredded().is_none());
        assert_variant_values(&variant, &[1, 2, 3])
    }

    #[test]
    fn pack_variant_chunks_all_shredded_same_dtype() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let chunked = ChunkedArray::try_new(
            vec![
                variant_chunk_with_shredded(
                    [1, 2],
                    PrimitiveArray::from_iter([10i32, 20]).into_array(),
                )?
                .into_array(),
                variant_chunk_with_shredded([3], PrimitiveArray::from_iter([30i32]).into_array())?
                    .into_array(),
            ],
            VariantDType(NonNullable),
        )?
        .into_array();

        let variant = into_variant(chunked.execute::<Canonical>(&mut ctx)?)?;
        let shredded = variant
            .shredded()
            .ok_or_else(|| vortex_err!("expected shredded child"))?;

        assert_eq!(shredded.dtype(), &Primitive(I32, NonNullable));
        assert_eq!(shredded.len(), 3);
        assert_variant_values(&variant, &[10, 20, 30])?;

        let shredded = shredded.clone().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(
            shredded,
            PrimitiveArray::from_iter([10i32, 20, 30]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn pack_variant_chunks_mixed_shredded_presence_errors() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let chunked = ChunkedArray::try_new(
            vec![
                variant_chunk_with_shredded([1], PrimitiveArray::from_iter([10i32]).into_array())?
                    .into_array(),
                variant_chunk([2])?.into_array(),
            ],
            VariantDType(NonNullable),
        )?
        .into_array();

        let err = chunked.execute::<Canonical>(&mut ctx).unwrap_err();
        assert!(
            err.to_string()
                .contains("chunks disagree on shredded presence")
        );
        Ok(())
    }

    #[test]
    fn pack_variant_chunks_mismatched_shredded_dtype_errors() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let chunked = ChunkedArray::try_new(
            vec![
                variant_chunk_with_shredded([1], PrimitiveArray::from_iter([10i32]).into_array())?
                    .into_array(),
                variant_chunk_with_shredded([2], PrimitiveArray::from_iter([20i64]).into_array())?
                    .into_array(),
            ],
            VariantDType(NonNullable),
        )?
        .into_array();

        let err = chunked.execute::<Canonical>(&mut ctx).unwrap_err();
        assert!(err.to_string().contains("shredded dtype mismatch"));
        Ok(())
    }

    #[test]
    fn pack_variant_chunks_empty() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let chunked = ChunkedArray::try_new(vec![], VariantDType(NonNullable))?.into_array();

        let variant = into_variant(chunked.execute::<Canonical>(&mut ctx)?)?;

        assert_eq!(variant.len(), 0);
        assert!(variant.shredded().is_none());
        Ok(())
    }

    #[test]
    fn pack_variant_chunks_single_chunk() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let chunked = ChunkedArray::try_new(
            vec![
                variant_chunk_with_shredded(
                    [1, 2],
                    PrimitiveArray::from_iter([10i32, 20]).into_array(),
                )?
                .into_array(),
            ],
            VariantDType(NonNullable),
        )?
        .into_array();

        let variant = into_variant(chunked.execute::<Canonical>(&mut ctx)?)?;

        assert_eq!(variant.len(), 2);
        assert!(variant.shredded().is_some());
        assert_variant_values(&variant, &[10, 20])
    }

    #[test]
    pub fn pack_nested_structs() {
        let mut ctx = SESSION.create_execution_ctx();
        let struct_array = StructArray::try_new(
            ["a"].into(),
            vec![VarBinViewArray::from_iter_str(["foo", "bar", "baz", "quak"]).into_array()],
            4,
            Validity::NonNullable,
        )
        .unwrap();
        let dtype = struct_array.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                ChunkedArray::try_new(vec![struct_array.clone().into_array()], dtype.clone())
                    .unwrap()
                    .into_array(),
            ],
            dtype,
        )
        .unwrap()
        .into_array();
        let canonical_struct = chunked.execute::<StructArray>(&mut ctx).unwrap();
        let canonical_varbin = canonical_struct
            .unmasked_field(0)
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)
            .unwrap();
        let original_varbin = struct_array
            .unmasked_field(0)
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)
            .unwrap();
        let orig_mask = original_varbin
            .validity()
            .unwrap()
            .execute_mask(original_varbin.len(), &mut ctx)
            .unwrap();
        let orig_values = (0..original_varbin.len())
            .map(|i| {
                orig_mask
                    .value(i)
                    .then(|| original_varbin.bytes_at(i).to_vec())
            })
            .collect::<Vec<_>>();
        let canon_mask = canonical_varbin
            .validity()
            .unwrap()
            .execute_mask(canonical_varbin.len(), &mut ctx)
            .unwrap();
        let canon_values = (0..canonical_varbin.len())
            .map(|i| {
                canon_mask
                    .value(i)
                    .then(|| canonical_varbin.bytes_at(i).to_vec())
            })
            .collect::<Vec<_>>();
        assert_eq!(orig_values, canon_values);
    }

    #[test]
    pub fn pack_nested_lists() {
        let mut ctx = SESSION.create_execution_ctx();
        let l1 = ListArray::try_new(
            buffer![1, 2, 3, 4].into_array(),
            buffer![0, 3].into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let l2 = ListArray::try_new(
            buffer![5, 6].into_array(),
            buffer![0, 2].into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let chunked_list = ChunkedArray::try_new(
            vec![l1.clone().into_array(), l2.clone().into_array()],
            List(Arc::new(Primitive(I32, NonNullable)), NonNullable),
        );

        let canon_values = chunked_list
            .unwrap()
            .as_array()
            .clone()
            .execute::<ListViewArray>(&mut ctx)
            .unwrap();

        assert_eq!(
            l1.execute_scalar(0, &mut ctx).unwrap(),
            canon_values.execute_scalar(0, &mut ctx).unwrap()
        );
        assert_eq!(
            l2.execute_scalar(0, &mut ctx).unwrap(),
            canon_values.execute_scalar(1, &mut ctx).unwrap()
        );
    }

    #[test]
    fn pack_fixed_size_lists() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let f1 = FixedSizeListArray::try_new(
            buffer![1, 2, 3, 4, 5, 6].into_array(),
            2,
            Validity::NonNullable,
            3,
        )?;
        let f2 = FixedSizeListArray::try_new(
            buffer![7, 8, 9, 10].into_array(),
            2,
            Validity::NonNullable,
            2,
        )?;
        let dtype = f1.dtype().clone();

        let chunked =
            ChunkedArray::try_new(vec![f1.into_array(), f2.into_array()], dtype)?.into_array();

        let canonical = chunked.clone().execute::<Canonical>(&mut ctx)?;
        let fsl = match canonical {
            Canonical::FixedSizeList(fsl) => fsl,
            other => vortex_bail!("expected FixedSizeList canonical array, got {other:?}"),
        };

        assert_eq!(fsl.len(), 5);
        let expected = FixedSizeListArray::try_new(
            buffer![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array(),
            2,
            Validity::NonNullable,
            5,
        )?;
        for idx in 0..5 {
            assert_eq!(
                chunked.execute_scalar(idx, &mut ctx)?,
                expected.execute_scalar(idx, &mut ctx)?,
            );
        }
        Ok(())
    }

    /// Canonicalizing a `ChunkedArray` reuses each chunk's children instead of concatenating
    /// them: a nested builder keeps a child chunked on exactly the boundaries it was appended on,
    /// however short the appended chunk is.
    #[rstest]
    #[case::struct_(
        StructArray::try_from_iter([("a", buffer![1i32, 2])])
            .vortex_expect("struct array")
            .into_array(),
        |array: &ArrayRef| array.as_::<Struct>().unmasked_field(0).clone()
    )]
    #[case::fixed_size_list(
        FixedSizeListArray::new(buffer![1i32, 2].into_array(), 2, Validity::NonNullable, 1)
            .into_array(),
        |array: &ArrayRef| array.as_::<FixedSizeList>().elements().clone()
    )]
    #[case::list(
        ListArray::try_new(
            buffer![1i32, 2].into_array(),
            buffer![0, 2].into_array(),
            Validity::NonNullable,
        )
            .vortex_expect("list array")
            .into_array(),
        |array: &ArrayRef| array.as_::<ListView>().elements().clone()
    )]
    fn canonicalize_reuses_chunk_children(
        #[case] chunk: ArrayRef,
        #[case] child_of: fn(&ArrayRef) -> ArrayRef,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let chunked =
            ChunkedArray::try_new(vec![chunk.clone(), chunk.clone()], chunk.dtype().clone())?
                .into_array();
        let canonical = chunked.execute::<Canonical>(&mut ctx)?.into_array();

        let child = child_of(&canonical);
        assert_eq!(
            child.as_::<Chunked>().nchunks(),
            2,
            "each chunk's child should have become a chunk of the combined child",
        );

        let expected =
            ChunkedArray::try_new(vec![chunk.clone(), chunk], canonical.dtype().clone())?;
        assert_arrays_eq!(&canonical, &expected, &mut ctx);

        Ok(())
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
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
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VariantArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewRebuildMode;
use crate::arrays::variant::VariantArrayExt;
use crate::builders::builder_with_capacity_in;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::memory::HostAllocatorExt;
use crate::validity::Validity;

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

    let owned_chunks: Vec<ArrayRef> = array.iter_chunks().cloned().collect();
    Ok(match array.dtype() {
        DType::Struct(..) => {
            let struct_array = pack_struct_chunks(owned_chunks, ctx)?;
            Canonical::Struct(struct_array)
        }
        DType::List(elem_dtype, _) => Canonical::List(swizzle_list_chunks(
            &owned_chunks,
            array.array().validity()?,
            elem_dtype,
            ctx,
        )?),
        DType::FixedSizeList(elem_dtype, list_size, _) => {
            Canonical::FixedSizeList(swizzle_fixed_size_list_chunks(
                &owned_chunks,
                array.array().validity()?,
                elem_dtype,
                *list_size,
                ctx,
            )?)
        }
        DType::Variant(_) => Canonical::Variant(pack_variant_chunks(owned_chunks, ctx)?),
        _ => {
            let mut builder = builder_with_capacity_in(ctx.allocator(), array.dtype(), array.len());
            array.array().append_to_builder(builder.as_mut(), ctx)?;
            builder.finish_into_canonical()
        }
    })
}

/// Packs many [`StructArray`]s to instead be a single [`StructArray`], where the [`DynArrayData`] for each
/// field is a [`ChunkedArray`].
///
/// The caller guarantees there are at least 2 chunks.
fn pack_struct_chunks(chunks: Vec<ArrayRef>, ctx: &mut ExecutionCtx) -> VortexResult<StructArray> {
    chunks
        .into_iter()
        .map(|c| c.execute::<StructArray>(ctx))
        .process_results(|iter| StructArray::try_concat(iter))?
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
    let core_chunks = variant_chunks
        .iter()
        .map(|chunk| chunk.core_storage().clone())
        .collect();
    let core_storage = ChunkedArray::try_new(core_chunks, outer_dtype)?.into_array();

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

/// Packs [`ListViewArray`]s together into a chunked `ListViewArray`.
///
/// We use the existing arrays (chunks) to form a chunked array of `elements` (the child array).
///
/// The caller guarantees there are at least 2 chunks.
fn swizzle_list_chunks(
    chunks: &[ArrayRef],
    validity: Validity,
    elem_dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ListViewArray> {
    let len: usize = chunks.iter().map(|c| c.len()).sum();

    assert_eq!(
        chunks[0]
            .dtype()
            .as_list_element_opt()
            .vortex_expect("DType was somehow not a list")
            .as_ref(),
        elem_dtype
    );

    // Since each list array in `chunks` has offsets local to each array, we can reuse the existing
    // array's child `elements` as the chunks and recompute offsets.

    let mut list_elements_chunks = Vec::with_capacity(chunks.len());
    let mut num_elements = 0;

    // TODO(connor)[ListView]: We could potentially choose a smaller type here, but that would make
    // this much more complicated.
    // We (somewhat arbitrarily) choose `u64` for our offsets and sizes here. These can always be
    // narrowed later by the compressor.
    let allocator = ctx.allocator();
    let mut offsets = allocator.allocate_typed::<u64>(len)?;
    let mut sizes = allocator.allocate_typed::<u64>(len)?;
    let offsets_out: &mut [u64] = offsets.as_mut_slice_typed::<u64>()?;
    let sizes_slice_out: &mut [u64] = sizes.as_mut_slice_typed::<u64>()?;
    let mut next_list = 0usize;

    for chunk in chunks {
        let chunk_array = chunk.clone().execute::<ListViewArray>(ctx)?;
        // By rebuilding as zero-copy to `List` and trimming all elements (to prevent gaps), we make
        // the final output `ListView` also zero-copyable to `List`.
        let chunk_array = chunk_array.rebuild(ListViewRebuildMode::MakeExact, ctx)?;

        // Add the `elements` of the current array as a new chunk.
        list_elements_chunks.push(chunk_array.elements().clone());

        // Cast offsets and sizes to `u64`.
        let offsets_arr = chunk_array
            .offsets()
            .clone()
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))
            .vortex_expect("Must be able to fit array offsets in u64")
            .execute::<PrimitiveArray>(ctx)?;

        let sizes_arr = chunk_array
            .sizes()
            .clone()
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))
            .vortex_expect("Must be able to fit array offsets in u64")
            .execute::<PrimitiveArray>(ctx)?;

        let offsets_slice = offsets_arr.as_slice::<u64>();
        let sizes_slice = sizes_arr.as_slice::<u64>();

        // Append offsets and sizes, adjusting offsets to point into the combined array.
        for (&offset, &size) in offsets_slice.iter().zip(sizes_slice.iter()) {
            offsets_out[next_list] = offset + num_elements;
            sizes_slice_out[next_list] = size;
            next_list += 1;
        }

        num_elements += chunk_array.elements().len() as u64;
    }
    debug_assert_eq!(next_list, len);

    // SAFETY: elements are sliced from valid `ListViewArray`s (from `to_listview()`).
    let chunked_elements =
        unsafe { ChunkedArray::new_unchecked(list_elements_chunks, elem_dtype.clone()) }
            .into_array();

    let offsets = PrimitiveArray::new(
        Buffer::<u64>::from_byte_buffer(offsets.freeze()),
        Validity::NonNullable,
    )
    .into_array();
    let sizes = PrimitiveArray::new(
        Buffer::<u64>::from_byte_buffer(sizes.freeze()),
        Validity::NonNullable,
    )
    .into_array();

    // SAFETY:
    // - `offsets` and `sizes` are non-nullable u64 arrays of the same length
    // - Each `offset[i] + size[i]` list view is within bounds of elements array because it came
    //   from valid chunks
    // - Validity came from the outer chunked array so it must have the same length
    // - Since we made sure that all chunks were zero-copyable to a list above, we know that the
    //   final concatenated output is also zero-copyable to a list.
    Ok(unsafe {
        ListViewArray::new_unchecked(chunked_elements, offsets, sizes, validity)
            .with_zero_copy_to_list(true)
    })
}

/// Packs [`FixedSizeListArray`]s together into a single [`FixedSizeListArray`] whose `elements`
/// child is a [`ChunkedArray`].
///
/// Every chunk shares the same `list_size`, and each chunk's `elements` child is exactly
/// `list_size * chunk.len()` long and starts at the first list, so we can reuse the chunks'
/// `elements` children directly as the chunks of a combined `elements` array without copying.
///
/// The caller guarantees there are at least 2 chunks.
fn swizzle_fixed_size_list_chunks(
    chunks: &[ArrayRef],
    validity: Validity,
    elem_dtype: &DType,
    list_size: u32,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FixedSizeListArray> {
    let len: usize = chunks.iter().map(|c| c.len()).sum();

    let mut element_chunks = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let chunk_array = chunk.clone().execute::<FixedSizeListArray>(ctx)?;
        // A canonical `FixedSizeListArray` keeps its `elements` child trimmed to exactly
        // `list_size * chunk.len()` starting at the first list, so the children concatenate
        // cleanly into the combined `elements` array.
        element_chunks.push(chunk_array.elements().clone());
    }

    let chunked_elements = ChunkedArray::try_new(element_chunks, elem_dtype.clone())?.into_array();

    FixedSizeListArray::try_new(chunked_elements, list_size, validity, len)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::accessor::ArrayAccessor;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::VariantArray;
    use crate::arrays::struct_::StructArrayExt;
    use crate::arrays::variant::VariantArrayExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType::List;
    use crate::dtype::DType::Primitive;
    use crate::dtype::DType::Variant as VariantDType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType::I32;
    use crate::memory::DefaultHostAllocator;
    use crate::memory::HostAllocator;
    use crate::memory::MemorySessionExt;
    use crate::memory::WritableHostBuffer;
    use crate::scalar::Scalar;
    use crate::session::ArraySession;
    use crate::validity::Validity;

    /// A shared session for these chunked-array tests, used to create execution contexts.
    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[derive(Debug)]
    struct CountingAllocator {
        allocations: Arc<AtomicUsize>,
    }

    impl HostAllocator for CountingAllocator {
        fn allocate(
            &self,
            len: usize,
            alignment: vortex_buffer::Alignment,
        ) -> VortexResult<WritableHostBuffer> {
            self.allocations.fetch_add(1, Ordering::Relaxed);
            DefaultHostAllocator.allocate(len, alignment)
        }
    }

    fn variant_scalar(value: i32) -> Scalar {
        Scalar::variant(Scalar::primitive(value, NonNullable))
    }

    fn variant_core(values: impl IntoIterator<Item = i32>) -> VortexResult<ArrayRef> {
        let chunks = values
            .into_iter()
            .map(|value| ConstantArray::new(variant_scalar(value), 1).into_array())
            .collect();

        Ok(ChunkedArray::try_new(chunks, VariantDType(NonNullable))?.into_array())
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
        assert_arrays_eq!(shredded, PrimitiveArray::from_iter([10i32, 20, 30]));
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
        let orig_values = original_varbin
            .with_iterator(|it| it.map(|a| a.map(|v| v.to_vec())).collect::<Vec<_>>());
        let canon_values = canonical_varbin
            .with_iterator(|it| it.map(|a| a.map(|v| v.to_vec())).collect::<Vec<_>>());
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

    #[test]
    fn list_canonicalize_uses_memory_session_allocator() {
        let allocations = Arc::new(AtomicUsize::new(0));
        let session = VortexSession::empty();
        session
            .memory_mut()
            .set_allocator(Arc::new(CountingAllocator {
                allocations: Arc::clone(&allocations),
            }));
        let mut ctx = ExecutionCtx::new(session);

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
            vec![l1.into_array(), l2.into_array()],
            List(Arc::new(Primitive(I32, NonNullable)), NonNullable),
        )
        .unwrap()
        .into_array();

        drop(chunked_list.execute::<Canonical>(&mut ctx).unwrap());
        assert!(
            allocations.load(Ordering::Relaxed) >= 2,
            "expected offset+size allocations through MemorySession"
        );
    }
}

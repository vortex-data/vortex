// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewRebuildMode;
use crate::arrays::struct_::StructArrayExt;
use crate::builders::builder_with_capacity_in;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::memory::HostAllocatorExt;
use crate::validity::Validity;

pub(super) fn _canonicalize(
    array: ArrayView<'_, Chunked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Canonical> {
    if array.nchunks() == 0 {
        return Ok(Canonical::empty(array.dtype()));
    }
    if array.nchunks() == 1 {
        return array.chunk(0).clone().execute::<Canonical>(ctx);
    }

    let owned_chunks: Vec<ArrayRef> = array.iter_chunks().cloned().collect();
    Ok(match array.dtype() {
        DType::Struct(struct_dtype, _) => {
            let struct_array =
                pack_struct_chunks(&owned_chunks, array.array().validity()?, struct_dtype, ctx)?;
            Canonical::Struct(struct_array)
        }
        DType::List(elem_dtype, _) => Canonical::List(swizzle_list_chunks(
            &owned_chunks,
            array.array().validity()?,
            elem_dtype,
            ctx,
        )?),
        _ => {
            let mut builder = builder_with_capacity_in(ctx.allocator(), array.dtype(), array.len());
            array.array().append_to_builder(builder.as_mut(), ctx)?;
            builder.finish_into_canonical()
        }
    })
}

/// Packs many [`StructArray`]s to instead be a single [`StructArray`], where the [`DynArray`] for each
/// field is a [`ChunkedArray`].
///
/// The caller guarantees there are at least 2 chunks.
fn pack_struct_chunks(
    chunks: &[ArrayRef],
    validity: Validity,
    struct_dtype: &StructFields,
    ctx: &mut ExecutionCtx,
) -> VortexResult<StructArray> {
    let len = chunks.iter().map(|chunk| chunk.len()).sum();
    let mut field_arrays = Vec::new();

    let executed_chunks: Vec<StructArray> = chunks
        .iter()
        .map(|c| c.clone().execute::<StructArray>(ctx))
        .collect::<VortexResult<_>>()?;

    for (field_idx, field_dtype) in struct_dtype.fields().enumerate() {
        let mut field_chunks = Vec::with_capacity(chunks.len());
        for struct_array in &executed_chunks {
            let field = struct_array.unmasked_field(field_idx).clone();
            field_chunks.push(field);
        }

        // SAFETY: field_chunks are extracted from valid StructArrays with matching dtypes.
        // Each chunk's field array is guaranteed to be valid for field_dtype.
        let field_array = unsafe { ChunkedArray::new_unchecked(field_chunks, field_dtype.clone()) };
        field_arrays.push(field_array.into_array());
    }

    // SAFETY: field_arrays are built from corresponding chunks of same length, dtypes match by
    // construction.
    Ok(unsafe { StructArray::new_unchecked(field_arrays, struct_dtype.clone(), len, validity) })
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
        let chunk_array = chunk_array.rebuild(ListViewRebuildMode::MakeExact)?;

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Canonical;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    #[expect(deprecated)]
    use crate::ToCanonical as _;
    use crate::VortexSessionExecute;
    use crate::accessor::ArrayAccessor;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ListArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::struct_::StructArrayExt;
    use crate::dtype::DType::List;
    use crate::dtype::DType::Primitive;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType::I32;
    use crate::memory::DefaultHostAllocator;
    use crate::memory::HostAllocator;
    use crate::memory::MemorySessionExt;
    use crate::memory::WritableHostBuffer;
    use crate::validity::Validity;

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

    #[test]
    pub fn pack_nested_structs() {
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
        #[expect(deprecated)]
        let canonical_struct = chunked.to_struct();
        #[expect(deprecated)]
        let canonical_varbin = canonical_struct.unmasked_field(0).to_varbinview();
        #[expect(deprecated)]
        let original_varbin = struct_array.unmasked_field(0).to_varbinview();
        let orig_values = original_varbin
            .with_iterator(|it| it.map(|a| a.map(|v| v.to_vec())).collect::<Vec<_>>());
        let canon_values = canonical_varbin
            .with_iterator(|it| it.map(|a| a.map(|v| v.to_vec())).collect::<Vec<_>>());
        assert_eq!(orig_values, canon_values);
    }

    #[test]
    pub fn pack_nested_lists() {
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

        #[expect(deprecated)]
        let canon_values = chunked_list.unwrap().as_array().to_listview();

        assert_eq!(
            l1.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            canon_values
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            l2.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            canon_values
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
        );
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

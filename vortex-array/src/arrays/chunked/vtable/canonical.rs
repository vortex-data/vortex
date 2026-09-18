// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::List;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::VariantArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::chunked::array::ChunkedSlots;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::variant::VariantArraySlotsExt;
use crate::dtype::DType;
use crate::match_each_integer_ptype;
use crate::matcher::Matcher;
use crate::validity::Validity;

pub(super) struct ListChunks;

impl Matcher for ListChunks {
    type Match<'a> = ();

    fn try_match(array: &ArrayRef) -> Option<()> {
        (array.is::<List>() || array.is::<ListView>()).then_some(())
    }
}

/// Executes chunks into the required encoding one at a time, then transposes their children.
pub(super) fn swizzle<V: Matcher + 'static>(
    array: ChunkedArray,
    finish: impl FnOnce(ChunkedArray) -> VortexResult<ArrayRef>,
) -> VortexResult<ExecutionResult> {
    let next_chunk = array
        .next_child_slot
        .saturating_sub(ChunkedSlots::CHUNKS_OFFSET);
    let needs_execution = array
        .iter_chunks()
        .enumerate()
        .skip(next_chunk)
        .find(|(_, chunk)| !V::matches(chunk))
        .map(|(idx, _)| idx + ChunkedSlots::CHUNKS_OFFSET);
    if let Some(slot) = needs_execution {
        return Ok(ExecutionResult::execute_slot::<V>(
            array.with_next_child_slot(slot + 1),
            slot,
        ));
    }

    Ok(ExecutionResult::done(finish(array)?))
}

pub(super) fn swizzle_fixed_size_list(array: ChunkedArray) -> VortexResult<ArrayRef> {
    let DType::FixedSizeList(element_dtype, list_size, _) = array.dtype() else {
        unreachable!("called only for a fixed-size list dtype")
    };
    // Canonical FSL children are trimmed to list_size * len; their elements concatenate directly.
    let element_chunks: Vec<_> = array
        .iter_chunks()
        .map(|chunk| chunk.as_::<FixedSizeList>().elements().clone())
        .collect();
    let validity = array.validity()?;
    // SAFETY: all chunks share the parent's FSL dtype, so their elements share element_dtype
    // and their lengths sum to list_size * array.len(). The parent supplies the combined validity.
    Ok(unsafe {
        let elements =
            ChunkedArray::new_unchecked(element_chunks, element_dtype.as_ref().clone())
                .into_array();
        FixedSizeListArray::new_unchecked(elements, *list_size, validity, array.len()).into_array()
    })
}

pub(super) fn swizzle_struct(array: ChunkedArray) -> VortexResult<ArrayRef> {
    let struct_fields = array.dtype().as_struct_fields();
    let chunks: Vec<_> = array
        .iter_chunks()
        .map(|chunk| chunk.as_::<Struct>())
        .collect();
    let fields = struct_fields.fields().enumerate().map(|(idx, dtype)| {
        let children = chunks.iter().map(|chunk| chunk.unmasked_field(idx).clone());
        // SAFETY: each chunk has the parent's struct dtype, which fixes each field's dtype.
        unsafe { ChunkedArray::new_unchecked(children, dtype) }.into_array()
    });
    let validity = array.validity()?;
    // SAFETY: each field concatenates the same chunk lengths, summing to array.len(). Their
    // dtypes come from the parent and its validity applies to exactly those rows, including nulls.
    Ok(unsafe {
        StructArray::new_unchecked(fields, struct_fields.clone(), array.len(), validity)
            .into_array()
    })
}

pub(super) fn swizzle_list(array: ChunkedArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let DType::List(element_dtype, _) = array.dtype() else {
        unreachable!("called only for a list dtype")
    };
    let mut elements = Vec::with_capacity(array.nchunks());
    let mut offsets = ctx.allocator().zeroed::<u64>(array.len());
    let mut sizes = ctx.allocator().zeroed::<u64>(array.len());
    let mut element_base = 0usize;
    let mut row = 0;
    let mut zero_copy_to_list = true;

    for chunk in array.iter_chunks() {
        if chunk.is_empty() {
            continue;
        }
        let offsets_out = &mut offsets.as_mut_slice()[row..row + chunk.len()];
        let sizes_out = &mut sizes.as_mut_slice()[row..row + chunk.len()];
        let (child, start, end, chunk_zero_copy_to_list) = if let Some(list) = chunk.as_opt::<List>() {
            let chunk_offsets = list.offsets().clone().execute::<PrimitiveArray>(ctx)?;
            match_each_integer_ptype!(chunk_offsets.ptype(), |O| {
                for ((out, size), pair) in offsets_out
                    .iter_mut()
                    .zip(sizes_out.iter_mut())
                    .zip(chunk_offsets.as_slice::<O>().windows(2))
                {
                    let start = u64::try_from(pair[0])
                        .vortex_expect("validated list offset is nonnegative");
                    let end = u64::try_from(pair[1])
                        .vortex_expect("validated list offset is nonnegative");
                    *out = start;
                    *size = end - start;
                }
            });
            let last = chunk.len() - 1;
            (
                list.elements().clone(),
                offsets_out[0],
                offsets_out[last] + sizes_out[last],
                true,
            )
        } else {
            let list = chunk.as_::<ListView>();
            let chunk_offsets = list.offsets().clone().execute::<PrimitiveArray>(ctx)?;
            let chunk_sizes = list.sizes().clone().execute::<PrimitiveArray>(ctx)?;
            match_each_integer_ptype!(chunk_offsets.ptype(), |O| {
                for (out, &offset) in offsets_out.iter_mut().zip(chunk_offsets.as_slice::<O>()) {
                    *out = u64::try_from(offset)
                        .vortex_expect("validated list offset is nonnegative");
                }
            });
            match_each_integer_ptype!(chunk_sizes.ptype(), |S| {
                for (out, &size) in sizes_out.iter_mut().zip(chunk_sizes.as_slice::<S>()) {
                    *out = u64::try_from(size).vortex_expect("validated list size is nonnegative");
                }
            });
            // Exact views bound their window with the first and last row. Other layouts need
            // both extrema so that overlaps and interior gaps remain unchanged.
            let (start, end) = if list.is_zero_copy_to_list() {
                let last = chunk.len() - 1;
                (offsets_out[0], offsets_out[last] + sizes_out[last])
            } else {
                offsets_out.iter().zip(sizes_out.iter()).fold(
                    (u64::MAX, 0),
                    |(start, end), (&offset, &size)| (start.min(offset), end.max(offset + size)),
                )
            };
            (
                list.elements().clone(),
                start,
                end,
                list.is_zero_copy_to_list(),
            )
        };
        let start = usize::try_from(start).vortex_expect("offset is bounded by elements.len()");
        let end = usize::try_from(end).vortex_expect("view end is bounded by elements.len()");
        let next_base = element_base
            .checked_add(end - start)
            .ok_or_else(|| vortex_err!("combined list elements length overflow"))?;
        for offset in offsets_out {
            *offset = (*offset - start as u64) + element_base as u64;
        }
        elements.push(child.slice(start..end)?);
        zero_copy_to_list &= chunk_zero_copy_to_list;
        element_base = next_base;
        row += chunk.len();
    }
    let validity = array.validity()?;
    // SAFETY: element dtypes come from the common list dtype. Offsets/sizes are non-nullable u64
    // arrays with one entry per row. Trimming and rebasing preserve each validated view's bounds,
    // and the checked combined length prevents overflow. Validity comes from the parent. If every
    // chunk is zero-copyable, trimming makes adjacent chunks meet without gaps or overlaps.
    Ok(unsafe {
        let elements =
            ChunkedArray::new_unchecked(elements, element_dtype.as_ref().clone()).into_array();
        let offsets =
            PrimitiveArray::new_unchecked(offsets.freeze(), Validity::NonNullable).into_array();
        let sizes = PrimitiveArray::new_unchecked(sizes.freeze(), Validity::NonNullable).into_array();
        ListViewArray::new_unchecked(elements, offsets, sizes, validity)
            .with_zero_copy_to_list(zero_copy_to_list)
            .into_array()
    })
}

pub(super) fn _canonicalize(
    array: ArrayView<'_, Chunked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Canonical> {
    vortex_ensure!(
        array.dtype().is_variant(),
        "only variant needs recursive swizzling"
    );

    if array.nchunks() == 0 {
        return VariantArray::try_new(array.array().clone().into_array(), None)
            .map(Canonical::Variant);
    }
    if array.nchunks() == 1 {
        return array.chunk(0).clone().execute::<Canonical>(ctx);
    }

    Ok(Canonical::Variant(pack_variant_chunks(
        array.iter_chunks(),
        ctx,
    )?))
}

/// Packs many [`VariantArray`]s into one [`VariantArray`] with chunked children.
///
/// The caller guarantees there are at least 2 chunks.
fn pack_variant_chunks<'a>(
    chunks: impl Iterator<Item = &'a ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<VariantArray> {
    let variant_chunks: Vec<VariantArray> = chunks
        .into_iter()
        .map(|chunk| chunk.clone().execute::<VariantArray>(ctx))
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
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VariantArray;
    use crate::arrays::variant::VariantArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType::Primitive;
    use crate::dtype::DType::Variant as VariantDType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType::I32;
    use crate::scalar::Scalar;

    /// A shared session for these chunked-array tests, used to create execution contexts.
    static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

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
}

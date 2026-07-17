// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::PiecewiseSequence;
use crate::arrays::PiecewiseSequenceArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::list::ListArrayExt;
use crate::arrays::piecewise_sequence::UnitMultiplierLengths;
use crate::arrays::piecewise_sequence::execute_unit_multiplier_index_arrays;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::PrimitiveBuilder;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::dtype::UnsignedPType;
use crate::executor::ExecutionCtx;
use crate::match_each_unsigned_integer_ptype;
use crate::match_smallest_offset_type;
use crate::validity::Validity;

// TODO(connor)[ListView]: Re-revert to the version where we simply convert to a `ListView` and call
// the `ListView::take` compute function once `ListView` is more stable.

impl TakeExecute for List {
    /// Take implementation for [`ListArray`].
    ///
    /// Unlike `ListView`, `ListArray` must rebuild the elements array to maintain its invariant
    /// that lists are stored contiguously and in-order (`offset[i+1] >= offset[i]`). Taking
    /// non-contiguous indices would violate this requirement.
    #[expect(clippy::cognitive_complexity)]
    fn take(
        array: ArrayView<'_, List>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(piecewise_indices) = indices.as_opt::<PiecewiseSequence>()
            && let Some(taken) = take_piecewise_sequence(array, piecewise_indices, indices, ctx)?
        {
            return Ok(Some(taken));
        }

        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let indices = indices.reinterpret_cast(indices.ptype().to_unsigned());
        let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
        // This is an over-approximation of the total number of elements in the resulting array.
        let total_approx = array.elements().len().saturating_mul(indices.len());

        match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
            match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
                match_smallest_offset_type!(total_approx, |OutputOffsetType| {
                    _take::<I, O, OutputOffsetType>(
                        array,
                        offsets.as_view(),
                        indices.as_view(),
                        ctx,
                    )
                    .map(Some)
                })
            })
        })
    }
}

fn _take<I: IntegerPType, O: IntegerPType, OutputOffsetType: IntegerPType>(
    array: ArrayView<'_, List>,
    offsets_array: ArrayView<'_, Primitive>,
    indices_array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let data_validity = array
        .list_validity()
        .execute_mask(array.as_ref().len(), ctx)?;
    let indices_validity = indices_array
        .validity()
        .vortex_expect("Failed to compute validity mask")
        .execute_mask(indices_array.as_ref().len(), ctx)?;

    if !indices_validity.all_true() || !data_validity.all_true() {
        return _take_nullable::<I, O, OutputOffsetType>(array, offsets_array, indices_array, ctx);
    }

    let offsets: &[O] = offsets_array.as_slice();
    let indices: &[I] = indices_array.as_slice();

    let mut new_offsets = PrimitiveBuilder::<OutputOffsetType>::with_capacity(
        Nullability::NonNullable,
        indices.len(),
    );
    let mut elements_to_take =
        PrimitiveBuilder::with_capacity(Nullability::NonNullable, 2 * indices.len());

    let mut current_offset = OutputOffsetType::zero();
    new_offsets.append_zero();

    for &data_idx in indices {
        let data_idx: usize = data_idx.as_();

        let start = offsets[data_idx];
        let stop = offsets[data_idx + 1];

        // Annoyingly, we can't turn (start..end) into a range, so we're doing that manually.
        //
        // We could convert start and end to usize, but that would impose a potentially
        // harder constraint - now we don't care if they fit into usize as long as their
        // difference does.
        let additional: usize = (stop - start).as_();

        // TODO(0ax1): optimize this
        elements_to_take.reserve_exact(additional);
        for i in 0..additional {
            elements_to_take.append_value(start + O::from_usize(i).vortex_expect("i < additional"));
        }
        current_offset +=
            OutputOffsetType::from_usize((stop - start).as_()).vortex_expect("offset conversion");
        new_offsets.append_value(current_offset);
    }

    let elements_to_take = elements_to_take.finish();
    let new_offsets = new_offsets.finish();

    let new_elements = array.elements().take(elements_to_take)?;

    Ok(ListArray::try_new(
        new_elements,
        new_offsets,
        array.validity()?.take(indices_array.array())?,
    )?
    .into_array())
}

fn take_piecewise_sequence(
    array: ArrayView<'_, List>,
    indices: ArrayView<'_, PiecewiseSequence>,
    indices_ref: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let data_validity = array
        .list_validity()
        .execute_mask(array.as_ref().len(), ctx)?;
    if !data_validity.all_true() {
        return Ok(None);
    }

    let Some((starts, lengths)) = execute_unit_multiplier_index_arrays(indices, ctx)? else {
        return Ok(None);
    };
    let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
    let output_len = indices_ref.len();

    let taken = match &lengths {
        UnitMultiplierLengths::Constant(length) => take_piecewise_sequence_constant_dispatch(
            array,
            &starts,
            *length,
            &offsets,
            indices_ref,
            output_len,
        )?,
        UnitMultiplierLengths::Array(lengths) => take_piecewise_sequence_lengths_dispatch(
            array,
            &starts,
            lengths,
            &offsets,
            indices_ref,
            output_len,
        )?,
    };
    Ok(Some(taken))
}

fn take_piecewise_sequence_constant_dispatch(
    array: ArrayView<'_, List>,
    starts: &PrimitiveArray,
    length: usize,
    offsets: &PrimitiveArray,
    indices_ref: &ArrayRef,
    output_len: usize,
) -> VortexResult<ArrayRef> {
    match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
        take_piecewise_sequence_constant_start_dispatch::<S>(
            array,
            starts,
            length,
            offsets,
            indices_ref,
            output_len,
        )
    })
}

fn take_piecewise_sequence_constant_start_dispatch<S>(
    array: ArrayView<'_, List>,
    starts: &PrimitiveArray,
    length: usize,
    offsets: &PrimitiveArray,
    indices_ref: &ArrayRef,
    output_len: usize,
) -> VortexResult<ArrayRef>
where
    S: UnsignedPType,
{
    match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
        take_piecewise_sequence_constant_length::<S, O>(
            array,
            starts.as_slice::<S>(),
            length,
            offsets.as_slice::<O>(),
            indices_ref,
            output_len,
        )
    })
}

fn take_piecewise_sequence_lengths_dispatch(
    array: ArrayView<'_, List>,
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    offsets: &PrimitiveArray,
    indices_ref: &ArrayRef,
    output_len: usize,
) -> VortexResult<ArrayRef> {
    match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
        take_piecewise_sequence_lengths_start_dispatch::<S>(
            array,
            starts,
            lengths,
            offsets,
            indices_ref,
            output_len,
        )
    })
}

fn take_piecewise_sequence_lengths_start_dispatch<S>(
    array: ArrayView<'_, List>,
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    offsets: &PrimitiveArray,
    indices_ref: &ArrayRef,
    output_len: usize,
) -> VortexResult<ArrayRef>
where
    S: UnsignedPType,
{
    match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
        take_piecewise_sequence_lengths_start_length_dispatch::<S, L>(
            array,
            starts,
            lengths,
            offsets,
            indices_ref,
            output_len,
        )
    })
}

fn take_piecewise_sequence_lengths_start_length_dispatch<S, L>(
    array: ArrayView<'_, List>,
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    offsets: &PrimitiveArray,
    indices_ref: &ArrayRef,
    output_len: usize,
) -> VortexResult<ArrayRef>
where
    S: UnsignedPType,
    L: UnsignedPType,
{
    match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
        take_piecewise_sequence_typed::<S, L, O>(
            array,
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
            offsets.as_slice::<O>(),
            indices_ref,
            output_len,
        )
    })
}

fn take_piecewise_sequence_constant_length<S, Offset>(
    array: ArrayView<'_, List>,
    starts: &[S],
    length: usize,
    offsets: &[Offset],
    indices_ref: &ArrayRef,
    output_len: usize,
) -> VortexResult<ArrayRef>
where
    S: UnsignedPType,
    Offset: UnsignedPType,
{
    let computed_len = starts
        .len()
        .checked_mul(length)
        .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;
    vortex_ensure!(
        computed_len == output_len,
        "PiecewiseSequenceArray expanded length {computed_len} does not match declared length {output_len}"
    );
    let total_elements =
        piecewise_list_elements_len_constant(array.elements().len(), offsets, starts, length)?;
    let validity = array.validity()?.take(indices_ref)?;

    match_smallest_offset_type!(total_elements, |OutputOffset| {
        let gathered = gather_piecewise_list_constant_length::<S, Offset, OutputOffset>(
            array.elements(),
            offsets,
            starts,
            length,
            output_len,
            total_elements,
        )?;

        // SAFETY: output offsets are rebuilt from valid monotonic source offsets; output elements
        // are exactly the gathered child ranges referenced by those offsets; validity has one bit
        // per output row.
        Ok(
            unsafe { ListArray::new_unchecked(gathered.elements, gathered.offsets, validity) }
                .into_array(),
        )
    })
}

fn take_piecewise_sequence_typed<S, L, Offset>(
    array: ArrayView<'_, List>,
    starts: &[S],
    lengths: &[L],
    offsets: &[Offset],
    indices_ref: &ArrayRef,
    output_len: usize,
) -> VortexResult<ArrayRef>
where
    S: UnsignedPType,
    L: UnsignedPType,
    Offset: UnsignedPType,
{
    let mut computed_len = 0usize;
    for &length in lengths {
        let length: usize = length.as_();
        computed_len = computed_len
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;
    }
    vortex_ensure!(
        computed_len == output_len,
        "PiecewiseSequenceArray expanded length {computed_len} does not match declared length {output_len}"
    );
    let total_elements =
        piecewise_list_elements_len(array.elements().len(), offsets, starts, lengths)?;

    match_smallest_offset_type!(total_elements, |OutputOffset| {
        let gathered = gather_piecewise_list::<S, L, Offset, OutputOffset>(
            array.elements(),
            offsets,
            starts,
            lengths,
            output_len,
            total_elements,
        )?;
        let validity = array.validity()?.take(indices_ref)?;

        // SAFETY: output offsets are rebuilt from valid monotonic source offsets; output elements
        // are exactly the gathered child ranges referenced by those offsets; validity has one bit
        // per output row.
        Ok(
            unsafe { ListArray::new_unchecked(gathered.elements, gathered.offsets, validity) }
                .into_array(),
        )
    })
}

struct GatheredList {
    elements: ArrayRef,
    offsets: ArrayRef,
}

fn piecewise_list_elements_len_constant<S, Offset>(
    elements_len: usize,
    offsets: &[Offset],
    starts: &[S],
    length: usize,
) -> VortexResult<usize>
where
    S: UnsignedPType,
    Offset: UnsignedPType,
{
    let mut total = 0usize;
    for &start in starts {
        let start: usize = start.as_();
        if length == 0 {
            continue;
        }

        let offset_range = &offsets[start..][..=length];
        let element_start: usize = offset_range[0].as_();
        let element_end: usize = offset_range[length].as_();
        vortex_ensure!(
            element_start <= element_end && element_end <= elements_len,
            "List offsets range {element_start}..{element_end} exceeds elements length {elements_len}",
        );
        total = total
            .checked_add(element_end - element_start)
            .ok_or_else(|| vortex_err!("List take output elements length overflow"))?;
    }
    Ok(total)
}

fn piecewise_list_elements_len<S, L, Offset>(
    elements_len: usize,
    offsets: &[Offset],
    starts: &[S],
    lengths: &[L],
) -> VortexResult<usize>
where
    S: UnsignedPType,
    L: UnsignedPType,
    Offset: UnsignedPType,
{
    let mut total = 0usize;
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start: usize = start.as_();
        let length: usize = length.as_();
        if length == 0 {
            continue;
        }

        let offset_range = &offsets[start..][..=length];
        let element_start: usize = offset_range[0].as_();
        let element_end: usize = offset_range[length].as_();
        vortex_ensure!(
            element_start <= element_end && element_end <= elements_len,
            "List offsets range {element_start}..{element_end} exceeds elements length {elements_len}",
        );
        total = total
            .checked_add(element_end - element_start)
            .ok_or_else(|| vortex_err!("List take output elements length overflow"))?;
    }
    Ok(total)
}

fn gather_piecewise_list_constant_length<S, Offset, OutputOffset>(
    elements: &ArrayRef,
    offsets: &[Offset],
    starts: &[S],
    length: usize,
    output_len: usize,
    total_elements: usize,
) -> VortexResult<GatheredList>
where
    S: UnsignedPType,
    Offset: UnsignedPType,
    OutputOffset: IntegerPType,
{
    let offsets_capacity = output_len
        .checked_add(1)
        .ok_or_else(|| vortex_err!("List take offsets length overflow"))?;
    let mut new_offsets = BufferMut::<OutputOffset>::with_capacity(offsets_capacity);
    let mut element_starts = BufferMut::<u64>::with_capacity(starts.len());
    let mut element_lengths = BufferMut::<u64>::with_capacity(starts.len());
    let mut output_elements = 0usize;

    new_offsets.push(OutputOffset::zero());
    for &start in starts {
        let start: usize = start.as_();
        if length == 0 {
            continue;
        }

        let offset_range = &offsets[start..][..=length];
        let element_start: usize = offset_range[0].as_();
        let element_end: usize = offset_range[length].as_();
        for &offset in &offset_range[1..] {
            let offset: usize = offset.as_();
            let relative = offset
                .checked_sub(element_start)
                .ok_or_else(|| vortex_err!("List offsets are not monotonic at offset {offset}"))?;
            let output_offset = output_elements
                .checked_add(relative)
                .ok_or_else(|| vortex_err!("List take output elements length overflow"))?;
            new_offsets.push(new_offset_value::<OutputOffset>(output_offset)?);
        }

        let element_length = element_end - element_start;
        element_starts.push(element_start as u64);
        element_lengths.push(element_length as u64);
        output_elements = output_elements
            .checked_add(element_length)
            .ok_or_else(|| vortex_err!("List take output elements length overflow"))?;
    }
    debug_assert_eq!(output_elements, total_elements);

    let offsets = PrimitiveArray::new(new_offsets.freeze(), Validity::NonNullable).into_array();
    let multipliers = ConstantArray::new(1u64, element_starts.len()).into_array();
    // SAFETY: element ranges are derived from validated source list offsets, and total_elements is
    // the sum of the gathered element range lengths. Multiplier 1 preserves contiguous ranges.
    let element_indices = unsafe {
        PiecewiseSequenceArray::new_unchecked(
            element_starts.into_array(),
            element_lengths.into_array(),
            multipliers,
            total_elements,
        )
    };
    let elements = elements.take(element_indices.into_array())?;

    Ok(GatheredList { elements, offsets })
}

fn gather_piecewise_list<S, L, Offset, OutputOffset>(
    elements: &ArrayRef,
    offsets: &[Offset],
    starts: &[S],
    lengths: &[L],
    output_len: usize,
    total_elements: usize,
) -> VortexResult<GatheredList>
where
    S: UnsignedPType,
    L: UnsignedPType,
    Offset: UnsignedPType,
    OutputOffset: IntegerPType,
{
    let offsets_capacity = output_len
        .checked_add(1)
        .ok_or_else(|| vortex_err!("List take offsets length overflow"))?;
    let mut new_offsets = BufferMut::<OutputOffset>::with_capacity(offsets_capacity);
    let mut element_starts = BufferMut::<u64>::with_capacity(starts.len());
    let mut element_lengths = BufferMut::<u64>::with_capacity(lengths.len());
    let mut output_elements = 0usize;

    new_offsets.push(OutputOffset::zero());
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start: usize = start.as_();
        let length: usize = length.as_();
        if length == 0 {
            continue;
        }

        let offset_range = &offsets[start..][..=length];
        let element_start: usize = offset_range[0].as_();
        let element_end: usize = offset_range[length].as_();
        for &offset in &offset_range[1..] {
            let offset: usize = offset.as_();
            let relative = offset
                .checked_sub(element_start)
                .ok_or_else(|| vortex_err!("List offsets are not monotonic at offset {offset}"))?;
            let output_offset = output_elements
                .checked_add(relative)
                .ok_or_else(|| vortex_err!("List take output elements length overflow"))?;
            new_offsets.push(new_offset_value::<OutputOffset>(output_offset)?);
        }

        let element_length = element_end - element_start;
        element_starts.push(element_start as u64);
        element_lengths.push(element_length as u64);
        output_elements = output_elements
            .checked_add(element_length)
            .ok_or_else(|| vortex_err!("List take output elements length overflow"))?;
    }
    debug_assert_eq!(output_elements, total_elements);

    let offsets = PrimitiveArray::new(new_offsets.freeze(), Validity::NonNullable).into_array();
    let multipliers = ConstantArray::new(1u64, element_starts.len()).into_array();
    // SAFETY: element ranges are derived from validated source list offsets, and total_elements is
    // the sum of the gathered element range lengths. Multiplier 1 preserves contiguous ranges.
    let element_indices = unsafe {
        PiecewiseSequenceArray::new_unchecked(
            element_starts.into_array(),
            element_lengths.into_array(),
            multipliers,
            total_elements,
        )
    };
    let elements = elements.take(element_indices.into_array())?;

    Ok(GatheredList { elements, offsets })
}

fn new_offset_value<T: IntegerPType>(value: usize) -> VortexResult<T> {
    T::from_usize(value).ok_or_else(|| {
        vortex_err!(
            "List take offset value {value} does not fit in {}",
            T::PTYPE
        )
    })
}

// Kept out-of-line: as a single-callsite generic helper it would otherwise be inlined into every
// monomorphization of `_take`, duplicating the entire nullable path across all specializations.
#[inline(never)]
fn _take_nullable<I: IntegerPType, O: IntegerPType, OutputOffsetType: IntegerPType>(
    array: ArrayView<'_, List>,
    offsets_array: ArrayView<'_, Primitive>,
    indices_array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let offsets: &[O] = offsets_array.as_slice();
    let indices: &[I] = indices_array.as_slice();
    let data_validity = array
        .list_validity()
        .execute_mask(array.as_ref().len(), ctx)?;
    let indices_validity = indices_array
        .validity()
        .vortex_expect("Failed to compute validity mask")
        .execute_mask(indices_array.as_ref().len(), ctx)?;

    let mut new_offsets = PrimitiveBuilder::<OutputOffsetType>::with_capacity(
        Nullability::NonNullable,
        indices.len(),
    );

    // This will be the indices we push down to the child array to call `take` with.
    //
    // There are 2 things to note here:
    // - We do not know how many elements we need to take from our child since lists are variable
    //   size: thus we arbitrarily choose a capacity of `2 * # of indices`.
    // - The type of the primitive builder needs to fit the largest offset of the (parent)
    //   `ListArray`, so we make this `PrimitiveBuilder` generic over `O` (instead of `I`).
    let mut elements_to_take =
        PrimitiveBuilder::<O>::with_capacity(Nullability::NonNullable, 2 * indices.len());

    let mut current_offset = OutputOffsetType::zero();
    new_offsets.append_zero();

    for (data_idx, index_valid) in indices.iter().zip(indices_validity.iter()) {
        if !index_valid {
            new_offsets.append_value(current_offset);
            continue;
        }

        let data_idx: usize = data_idx.as_();

        if !data_validity.value(data_idx) {
            new_offsets.append_value(current_offset);
            continue;
        }

        let start = offsets[data_idx];
        let stop = offsets[data_idx + 1];

        // See the note in `_take` on the reasoning.
        let additional: usize = (stop - start).as_();

        elements_to_take.reserve_exact(additional);
        for i in 0..additional {
            elements_to_take.append_value(start + O::from_usize(i).vortex_expect("i < additional"));
        }
        current_offset +=
            OutputOffsetType::from_usize((stop - start).as_()).vortex_expect("offset conversion");
        new_offsets.append_value(current_offset);
    }

    let elements_to_take = elements_to_take.finish();
    let new_offsets = new_offsets.finish();
    let new_elements = array.elements().take(elements_to_take)?;

    Ok(ListArray::try_new(
        new_elements,
        new_offsets,
        array.validity()?.take(indices_array.array())?,
    )?
    .into_array())
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::IntoArray as _;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PiecewiseSequenceArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType::I32;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn nullable_take() {
        let mut ctx = array_session().create_execution_ctx();
        let list = ListArray::try_new(
            buffer![0i32, 5, 3, 4].into_array(),
            buffer![0, 2, 3, 4, 4].into_array(),
            Validity::Array(BoolArray::from_iter(vec![true, true, false, true]).into_array()),
        )
        .unwrap()
        .into_array();

        let idx =
            PrimitiveArray::from_option_iter(vec![Some(0), None, Some(1), Some(3)]).into_array();

        let result = list.take(idx).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::List(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                Nullability::Nullable
            )
        );

        let result = result.execute::<ListViewArray>(&mut ctx).unwrap();

        assert_eq!(result.len(), 4);

        let element_dtype: Arc<DType> = Arc::new(I32.into());

        assert!(
            result
                .is_valid(0, &mut array_session().create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            result
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::list(
                Arc::clone(&element_dtype),
                vec![0i32.into(), 5.into()],
                Nullability::Nullable
            )
        );

        assert!(
            result
                .is_invalid(1, &mut array_session().create_execution_ctx())
                .unwrap()
        );

        assert!(
            result
                .is_valid(2, &mut array_session().create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            result
                .execute_scalar(2, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::list(
                Arc::clone(&element_dtype),
                vec![3i32.into()],
                Nullability::Nullable
            )
        );

        assert!(
            result
                .is_valid(3, &mut array_session().create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            result
                .execute_scalar(3, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::list(element_dtype, vec![], Nullability::Nullable)
        );
    }

    #[test]
    fn change_validity() {
        let list = ListArray::try_new(
            buffer![0i32, 5, 3, 4].into_array(),
            buffer![0, 2, 3].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        let idx = PrimitiveArray::from_option_iter(vec![Some(0), Some(1), None]).into_array();
        // since idx is nullable, the final list will also be nullable

        let result = list.take(idx).unwrap();
        assert_eq!(
            result.dtype(),
            &DType::List(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                Nullability::Nullable
            )
        );
    }

    #[test]
    fn non_nullable_take() {
        let mut ctx = array_session().create_execution_ctx();
        let list = ListArray::try_new(
            buffer![0i32, 5, 3, 4].into_array(),
            buffer![0, 2, 3, 3, 4].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        let idx = buffer![1, 0, 2].into_array();

        let result = list.take(idx).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::List(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                Nullability::NonNullable
            )
        );

        let result = result.execute::<ListViewArray>(&mut ctx).unwrap();

        assert_eq!(result.len(), 3);

        let element_dtype: Arc<DType> = Arc::new(I32.into());

        assert!(
            result
                .is_valid(0, &mut array_session().create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            result
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::list(
                Arc::clone(&element_dtype),
                vec![3i32.into()],
                Nullability::NonNullable
            )
        );

        assert!(
            result
                .is_valid(1, &mut array_session().create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            result
                .execute_scalar(1, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::list(
                Arc::clone(&element_dtype),
                vec![0i32.into(), 5.into()],
                Nullability::NonNullable
            )
        );

        assert!(
            result
                .is_valid(2, &mut array_session().create_execution_ctx())
                .unwrap()
        );
        assert_eq!(
            result
                .execute_scalar(2, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::list(element_dtype, vec![], Nullability::NonNullable)
        );
    }

    #[test]
    fn piecewise_sequence_take() {
        let mut ctx = array_session().create_execution_ctx();
        let list = ListArray::try_new(
            buffer![0i32, 1, 2, 3, 4, 5, 6].into_array(),
            buffer![0u32, 2, 5, 5, 7].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();
        let idx = PiecewiseSequenceArray::try_new(
            buffer![1u64, 0].into_array(),
            buffer![2u64, 1].into_array(),
            ConstantArray::new(1u64, 2).into_array(),
            3,
        )
        .unwrap()
        .into_array();

        let result = list
            .take(idx)
            .unwrap()
            .execute::<ListViewArray>(&mut ctx)
            .unwrap();

        let element_dtype: Arc<DType> = Arc::new(I32.into());
        assert_eq!(
            result.execute_scalar(0, &mut ctx).unwrap(),
            Scalar::list(
                Arc::clone(&element_dtype),
                vec![2i32.into(), 3.into(), 4.into()],
                Nullability::NonNullable
            )
        );
        assert_eq!(
            result.execute_scalar(1, &mut ctx).unwrap(),
            Scalar::list(Arc::clone(&element_dtype), vec![], Nullability::NonNullable)
        );
        assert_eq!(
            result.execute_scalar(2, &mut ctx).unwrap(),
            Scalar::list(
                element_dtype,
                vec![0i32.into(), 1.into()],
                Nullability::NonNullable
            )
        );
    }

    #[test]
    fn test_take_empty_array() {
        let list = ListArray::try_new(
            buffer![0i32, 5, 3, 4].into_array(),
            buffer![0].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        let idx = PrimitiveArray::empty::<i32>(Nullability::Nullable).into_array();

        let result = list.take(idx).unwrap();
        assert_eq!(
            result.dtype(),
            &DType::List(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                Nullability::Nullable
            )
        );
        assert_eq!(result.len(), 0,);
    }

    #[rstest]
    #[case(ListArray::try_new(
        buffer![0i32, 1, 2, 3, 4, 5].into_array(),
        buffer![0, 2, 3, 5, 5, 6].into_array(),
        Validity::NonNullable,
    ).unwrap())]
    #[case(ListArray::try_new(
        buffer![10i32, 20, 30, 40, 50].into_array(),
        buffer![0, 2, 3, 4, 5].into_array(),
        Validity::Array(BoolArray::from_iter(vec![true, false, true, true]).into_array()),
    ).unwrap())]
    #[case(ListArray::try_new(
        buffer![1i32, 2, 3].into_array(),
        buffer![0, 0, 2, 2, 3].into_array(), // First and third are empty
        Validity::NonNullable,
    ).unwrap())]
    #[case(ListArray::try_new(
        buffer![42i32, 43].into_array(),
        buffer![0, 2].into_array(),
        Validity::NonNullable,
    ).unwrap())]
    #[case({
        let elements = buffer![0i32..200].into_array();
        let mut offsets = vec![0u64];
        for i in 1..=50 {
            offsets.push(offsets[i - 1] + (i as u64 % 5)); // Variable length lists
        }
        ListArray::try_new(
            elements,
            PrimitiveArray::from_iter(offsets).into_array(),
            Validity::NonNullable,
        ).unwrap()
    })]
    #[case(ListArray::try_new(
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None]).into_array(),
        buffer![0, 2, 3, 5].into_array(),
        Validity::NonNullable,
    ).unwrap())]
    fn test_take_list_conformance(#[case] list: ListArray) {
        test_take_conformance(
            &list.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }

    #[test]
    fn test_u64_offset_accumulation_non_nullable() {
        let mut ctx = array_session().create_execution_ctx();
        let elements = buffer![0i32; 200].into_array();
        let offsets = buffer![0u8, 200].into_array();
        let list = ListArray::try_new(elements, offsets, Validity::NonNullable)
            .unwrap()
            .into_array();

        // Take the same large list twice - would overflow u8 but works with u64.
        let idx = buffer![0u8, 0].into_array();
        let result = list.take(idx).unwrap();

        assert_eq!(result.len(), 2);

        let result_view = result.execute::<ListViewArray>(&mut ctx).unwrap();
        assert_eq!(result_view.len(), 2);
        assert!(
            result_view
                .is_valid(0, &mut array_session().create_execution_ctx())
                .unwrap()
        );
        assert!(
            result_view
                .is_valid(1, &mut array_session().create_execution_ctx())
                .unwrap()
        );
    }

    #[test]
    fn test_u64_offset_accumulation_nullable() {
        let mut ctx = array_session().create_execution_ctx();
        let elements = buffer![0i32; 150].into_array();
        let offsets = buffer![0u8, 150, 150].into_array();
        let validity = BoolArray::from_iter(vec![true, false]).into_array();
        let list = ListArray::try_new(elements, offsets, Validity::Array(validity))
            .unwrap()
            .into_array();

        // Take the same large list twice - would overflow u8 but works with u64.
        let idx = PrimitiveArray::from_option_iter(vec![Some(0u8), None, Some(0u8)]).into_array();
        let result = list.take(idx).unwrap();

        assert_eq!(result.len(), 3);

        let result_view = result.execute::<ListViewArray>(&mut ctx).unwrap();
        assert_eq!(result_view.len(), 3);
        assert!(
            result_view
                .is_valid(0, &mut array_session().create_execution_ctx())
                .unwrap()
        );
        assert!(
            result_view
                .is_invalid(1, &mut array_session().create_execution_ctx())
                .unwrap()
        );
        assert!(
            result_view
                .is_valid(2, &mut array_session().create_execution_ctx())
                .unwrap()
        );
    }

    /// Regression test for validity length mismatch bug.
    ///
    /// When source array has `Validity::Array(...)` and indices are non-nullable,
    /// the result validity must have length equal to indices.len(), not source.len().
    #[test]
    fn test_take_validity_length_mismatch_regression() {
        // Source array with explicit validity array (length 2).
        let list = ListArray::try_new(
            buffer![1i32, 2, 3, 4].into_array(),
            buffer![0, 2, 4].into_array(),
            Validity::Array(BoolArray::from_iter(vec![true, true]).into_array()),
        )
        .unwrap()
        .into_array();

        // Take more indices than source length (4 vs 2) with non-nullable indices.
        let idx = buffer![0u32, 1, 0, 1].into_array();

        // This should not panic - result should have length 4.
        let result = list.take(idx).unwrap();
        assert_eq!(result.len(), 4);
    }
}

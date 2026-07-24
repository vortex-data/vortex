// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PiecewiseSequenceArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builders::builder_with_capacity;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::optimizer::ArrayOptimizer;
use crate::validity::Validity;

/// Take implementation for [`FixedSizeListArray`].
///
/// `FixedSizeListArray` must rebuild its elements array because selected lists need to become
/// packed from offset 0. The FSL layer translates selected list rows into ordered element runs and
/// delegates the execution strategy to the elements child via `PiecewiseSequenceArray` indices.
impl TakeExecute for FixedSizeList {
    fn take(
        array: ArrayView<'_, FixedSizeList>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if array.is_empty() {
            return take_empty_fsl(array, indices, ctx).map(Some);
        }

        take_non_empty_fsl(array, indices, ctx).map(Some)
    }
}

fn take_empty_fsl(
    array: ArrayView<'_, FixedSizeList>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    debug_assert!(array.is_empty());

    let new_len = indices.len();
    if new_len != 0 {
        let indices_validity = indices.validity()?.execute_mask(new_len, ctx)?;
        vortex_ensure!(
            indices_validity.all_false(),
            "cannot take valid indices from an empty FixedSizeList"
        );
    }

    let list_size = array.list_size() as usize;
    let elements_len = new_len.checked_mul(list_size).ok_or_else(|| {
        vortex_err!(
            "FixedSizeList take output length overflow: {new_len} lists of size {list_size}"
        )
    })?;
    let new_elements = default_elements(array, elements_len);
    let new_validity = if new_len == 0 {
        array.validity()?.take(indices)?
    } else {
        Validity::AllInvalid
    };

    // SAFETY: empty output needs no child values; otherwise the index validity mask proves every
    // output row is null. Placeholder child elements have the exact length required by FSL.
    unsafe {
        FixedSizeListArray::new_unchecked(new_elements, array.list_size(), new_validity, new_len)
    }
    .into_array()
    .optimize_ctx(ctx.session())
}

fn take_non_empty_fsl(
    array: ArrayView<'_, FixedSizeList>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    debug_assert!(!array.is_empty());

    let DType::Primitive(ptype, _) = indices.dtype() else {
        vortex_bail!("Invalid indices dtype: {}", indices.dtype())
    };
    if !ptype.is_int() {
        vortex_bail!("Invalid indices dtype: {}", indices.dtype());
    }

    if array.list_size() == 0 {
        return take_non_empty_degenerate_fsl(array, indices, ctx);
    }

    let indices_array = indices.clone().execute::<PrimitiveArray>(ctx)?;
    match_each_integer_ptype!(indices_array.ptype(), |I| {
        take_non_empty_non_degenerate_fsl::<I>(array, indices, indices_array.as_view(), ctx)
    })
}

fn take_non_empty_degenerate_fsl(
    array: ArrayView<'_, FixedSizeList>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    debug_assert!(!array.is_empty());
    debug_assert_eq!(array.list_size(), 0);
    vortex_ensure!(
        array.elements().is_empty(),
        "degenerate list must have empty elements"
    );

    let indices_array = indices.clone().execute::<PrimitiveArray>(ctx)?;
    match_each_integer_ptype!(indices_array.ptype(), |I| {
        bounds_check_valid_indices::<I>(&indices_array.as_view(), array.as_ref().len(), ctx)
    })?;
    let new_validity = array.validity()?.take(indices)?;
    let new_len = indices_array.len();

    // SAFETY: degenerate FSL inputs have no elements, valid index payloads were checked against
    // the source length, and `Validity::take` produces validity for `new_len`.
    unsafe {
        FixedSizeListArray::new_unchecked(
            array.elements().clone(),
            array.list_size(),
            new_validity,
            new_len,
        )
    }
    .into_array()
    .optimize_ctx(ctx.session())
}

fn take_non_empty_non_degenerate_fsl<I: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices: &ArrayRef,
    indices_array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    debug_assert!(!array.is_empty());
    debug_assert_ne!(array.list_size(), 0);

    let (new_elements, new_len) =
        take_non_empty_non_degenerate_elements::<I>(array, indices_array, ctx)?;
    let new_validity = array.validity()?.take(indices)?;

    // SAFETY: `new_elements` has `new_len * list_size` elements, and `Validity::take` produces
    // validity for `new_len`.
    unsafe {
        FixedSizeListArray::new_unchecked(new_elements, array.list_size(), new_validity, new_len)
    }
    .into_array()
    .optimize_ctx(ctx.session())
}

fn take_non_empty_non_degenerate_elements<I: IntegerPType>(
    array: ArrayView<'_, FixedSizeList>,
    indices_array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ArrayRef, usize)> {
    debug_assert!(!array.is_empty());
    debug_assert_ne!(array.list_size(), 0);

    let list_size = array.list_size() as usize;
    let array_len = array.as_ref().len();
    let indices: &[I] = indices_array.as_slice::<I>();
    let new_len = indices.len();
    let elements_len = new_len.checked_mul(list_size).ok_or_else(|| {
        vortex_err!(
            "FixedSizeList take output length overflow: {new_len} lists of size {list_size}"
        )
    })?;

    let indices_validity = indices_array.validity()?.execute_mask(new_len, ctx)?;
    let starts = indices
        .iter()
        .zip_eq(indices_validity.iter())
        .map(|(&data_idx, is_index_valid)| {
            if !is_index_valid {
                return Ok(0);
            }

            let data_idx: usize = data_idx.as_();
            if data_idx >= array_len {
                vortex_bail!(OutOfBounds: data_idx, 0, array_len);
            }
            Ok((data_idx * list_size) as u64)
        })
        .process_results(|iter| iter.collect::<BufferMut<u64>>())?;

    let new_elements =
        take_element_runs(array.elements(), starts.freeze(), list_size, elements_len)?;

    Ok((new_elements, new_len))
}

fn bounds_check_valid_indices<I: IntegerPType>(
    indices_array: &ArrayView<'_, Primitive>,
    array_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let indices: &[I] = indices_array.as_slice::<I>();
    let indices_validity = indices_array.validity()?.execute_mask(indices.len(), ctx)?;

    for (&data_idx, is_index_valid) in indices.iter().zip_eq(indices_validity.iter()) {
        if is_index_valid {
            let data_idx = data_idx.as_();
            if data_idx >= array_len {
                vortex_bail!(OutOfBounds: data_idx, 0, array_len);
            }
        }
    }
    Ok(())
}

fn default_elements(array: ArrayView<'_, FixedSizeList>, len: usize) -> ArrayRef {
    let mut builder = builder_with_capacity(array.elements().dtype(), len);
    builder.append_defaults(len);
    builder.finish()
}

fn take_element_runs(
    elements: &ArrayRef,
    starts: Buffer<u64>,
    length: usize,
    output_len: usize,
) -> VortexResult<ArrayRef> {
    let run_count = starts.len();
    let starts = PrimitiveArray::new(starts, Validity::NonNullable).into_array();
    let lengths = ConstantArray::new(length as u64, run_count).into_array();
    let multipliers = ConstantArray::new(1u64, run_count).into_array();

    // SAFETY: callers produced one start per output row after validating list indices against the
    // source FSL length. `length` and multiplier 1 are represented as non-nullable unsigned
    // constant arrays, and `output_len` was computed as `run_count * length`.
    let indices =
        unsafe { PiecewiseSequenceArray::new_unchecked(starts, lengths, multipliers, output_len) };
    elements.take(indices.into_array())
}

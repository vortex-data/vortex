// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::Arc;

use itertools::Itertools as _;
use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PiecewiseSequence;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::piecewise_sequence::UnitMultiplierLengths;
use crate::arrays::piecewise_sequence::execute_unit_multiplier_index_arrays;
use crate::arrays::varbinview::BinaryView;
use crate::buffer::BufferHandle;
use crate::dtype::UnsignedPType;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::match_each_unsigned_integer_ptype;

impl TakeExecute for VarBinView {
    /// Take involves creating a new array that references the old array, just with the given set of views.
    fn take(
        array: ArrayView<'_, VarBinView>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(piecewise_indices) = indices.as_opt::<PiecewiseSequence>()
            && let Some(taken) = take_contiguous_ranges(array, piecewise_indices, indices, ctx)?
        {
            return Ok(Some(taken));
        }

        let validity = array.validity()?.take(indices)?;
        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;

        let indices_mask = indices
            .as_ref()
            .validity()?
            .execute_mask(indices.as_ref().len(), ctx)?;
        let views_buffer = match_each_integer_ptype!(indices.ptype(), |I| {
            take_views(array.views(), indices.as_slice::<I>(), &indices_mask)
        });

        // SAFETY: taking all components at same indices maintains invariants
        unsafe {
            Ok(Some(
                VarBinViewArray::new_handle_unchecked(
                    BufferHandle::new_host(views_buffer.into_byte_buffer()),
                    Arc::clone(array.data_buffers()),
                    array
                        .dtype()
                        .union_nullability(indices.dtype().nullability()),
                    validity,
                )
                .into_array(),
            ))
        }
    }
}

fn take_contiguous_ranges(
    array: ArrayView<'_, VarBinView>,
    indices: ArrayView<'_, PiecewiseSequence>,
    indices_ref: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some((starts, lengths)) = execute_unit_multiplier_index_arrays(indices, ctx)? else {
        return Ok(None);
    };
    let source = array.views();
    let output_len = indices_ref.len();
    let views = match &lengths {
        UnitMultiplierLengths::Constant(length) => {
            match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
                gather_view_slices_constant_length(
                    source,
                    starts.as_slice::<S>(),
                    *length,
                    output_len,
                )?
            })
        }
        UnitMultiplierLengths::Array(lengths) => {
            match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
                match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
                    gather_view_slices(
                        source,
                        starts.as_slice::<S>(),
                        lengths.as_slice::<L>(),
                        output_len,
                    )?
                })
            })
        }
    };
    let validity = array.validity()?.take(indices_ref)?;

    // SAFETY: ranges were validated against the source views, and copied views still reference the
    // same backing data buffers.
    unsafe {
        Ok(Some(
            VarBinViewArray::new_handle_unchecked(
                BufferHandle::new_host(views.into_byte_buffer()),
                Arc::clone(array.data_buffers()),
                array.dtype().clone(),
                validity,
            )
            .into_array(),
        ))
    }
}

fn take_views<I: AsPrimitive<usize>>(
    views_ref: &[BinaryView],
    indices: &[I],
    mask: &Mask,
) -> Buffer<BinaryView> {
    // NOTE(ngates): this deref is not actually trivial, so we run it once.
    // We do not use iter_bools directly, since the resulting dyn iterator cannot
    // implement TrustedLen.
    match mask.bit_buffer() {
        AllOr::All => {
            Buffer::<BinaryView>::from_trusted_len_iter(indices.iter().map(|i| views_ref[i.as_()]))
        }
        AllOr::None => Buffer::<BinaryView>::from_trusted_len_iter(iter::repeat_n(
            BinaryView::default(),
            indices.len(),
        )),
        AllOr::Some(buffer) => Buffer::<BinaryView>::from_trusted_len_iter(
            buffer.iter().zip(indices.iter()).map(|(valid, idx)| {
                if valid {
                    views_ref[idx.as_()]
                } else {
                    BinaryView::default()
                }
            }),
        ),
    }
}

fn gather_view_slices_constant_length<S>(
    source: &[BinaryView],
    starts: &[S],
    length: usize,
    output_len: usize,
) -> VortexResult<Buffer<BinaryView>>
where
    S: UnsignedPType,
{
    let computed_len = starts
        .len()
        .checked_mul(length)
        .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;
    vortex_ensure!(
        computed_len == output_len,
        "PiecewiseSequenceArray expanded length {computed_len} does not match declared length {output_len}"
    );

    let mut views = BufferMut::<BinaryView>::with_capacity(output_len);
    for &start in starts {
        let start = start.as_();
        let end = start
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray range overflows usize"))?;
        views.extend_from_slice(&source[start..end]);
    }

    Ok(views.freeze())
}

fn gather_view_slices<S, L>(
    source: &[BinaryView],
    starts: &[S],
    lengths: &[L],
    output_len: usize,
) -> VortexResult<Buffer<BinaryView>>
where
    S: UnsignedPType,
    L: UnsignedPType,
{
    let mut views = BufferMut::<BinaryView>::with_capacity(output_len);
    let mut computed_len = 0usize;
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = start.as_();
        let length = length.as_();
        let end = start
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray range overflows usize"))?;
        computed_len = computed_len
            .checked_add(length)
            .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;
        views.extend_from_slice(&source[start..end]);
    }

    vortex_ensure!(
        computed_len == output_len,
        "PiecewiseSequenceArray expanded length {computed_len} does not match declared length {output_len}"
    );
    Ok(views.freeze())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::varbinview::compute::take::PrimitiveArray;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::validity::Validity;

    #[test]
    fn take_nullable() -> VortexResult<()> {
        let arr = VarBinViewArray::from_iter_nullable_str([
            Some("one"),
            None,
            Some("three"),
            Some("four"),
            None,
            Some("six"),
        ]);

        let taken = arr.take(buffer![0, 3].into_array())?;

        assert!(taken.dtype().is_nullable());
        let mut ctx = array_session().create_execution_ctx();
        let taken = taken.execute::<VarBinViewArray>(&mut ctx)?;
        let mask = taken.validity()?.execute_mask(taken.len(), &mut ctx)?;
        let result = (0..taken.len())
            .map(|i| {
                mask.value(i)
                    .then(|| unsafe { String::from_utf8_unchecked(taken.bytes_at(i).to_vec()) })
            })
            .collect::<Vec<_>>();
        assert_eq!(result, [Some("one".to_string()), Some("four".to_string())]);
        Ok(())
    }

    #[test]
    fn take_nullable_indices() -> VortexResult<()> {
        let arr = VarBinViewArray::from_iter(["one", "two"].map(Some), DType::Utf8(NonNullable));

        let indices = PrimitiveArray::new(
            // Verify that garbage values at NULL indices are ignored.
            buffer![1u64, 999],
            Validity::from(BitBuffer::from(vec![true, false])),
        );

        let taken = arr.take(indices.into_array())?;

        assert!(taken.dtype().is_nullable());
        let mut ctx = array_session().create_execution_ctx();
        let taken = taken.execute::<VarBinViewArray>(&mut ctx)?;
        let mask = taken.validity()?.execute_mask(taken.len(), &mut ctx)?;
        let result = (0..taken.len())
            .map(|i| {
                mask.value(i)
                    .then(|| unsafe { String::from_utf8_unchecked(taken.bytes_at(i).to_vec()) })
            })
            .collect::<Vec<_>>();
        assert_eq!(result, [Some("two".to_string()), None]);
        Ok(())
    }

    #[rstest]
    #[case(VarBinViewArray::from_iter(
        ["hello", "world", "test", "data", "array"].map(Some),
        DType::Utf8(NonNullable),
    ))]
    #[case(VarBinViewArray::from_iter_nullable_str([
        Some("hello"),
        None,
        Some("test"),
        Some("data"),
        None,
    ]))]
    #[case(VarBinViewArray::from_iter(
        [b"hello".as_slice(), b"world", b"test", b"data", b"array"].map(Some),
        DType::Binary(NonNullable),
    ))]
    #[case(VarBinViewArray::from_iter(["single"].map(Some), DType::Utf8(NonNullable)))]
    fn test_take_varbinview_conformance(#[case] array: VarBinViewArray) {
        test_take_conformance(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}

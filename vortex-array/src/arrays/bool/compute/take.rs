// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use num_traits::AsPrimitive;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BitBufferView;
use vortex_buffer::get_bit;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PiecewiseSequence;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::dict::TakeExecute;
use crate::arrays::piecewise_sequence::execute_unit_multiplier_index_arrays;
use crate::arrays::piecewise_sequence::validate_index_ranges;
use crate::builtins::ArrayBuiltins;
use crate::dtype::UnsignedPType;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;

impl TakeExecute for Bool {
    fn take(
        array: ArrayView<'_, Bool>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(piecewise_indices) = indices.as_opt::<PiecewiseSequence>()
            && let Some(taken) = take_piecewise_sequence(array, piecewise_indices, indices, ctx)?
        {
            return Ok(Some(taken));
        }

        let indices_nulls_zeroed = match indices.validity()?.execute_mask(indices.len(), ctx)? {
            Mask::AllTrue(_) => indices.clone(),
            Mask::AllFalse(_) => {
                return Ok(Some(
                    ConstantArray::new(Scalar::null(array.dtype().as_nullable()), indices.len())
                        .into_array(),
                ));
            }
            Mask::Values(_) => indices
                .clone()
                .fill_null(Scalar::from(0).cast(indices.dtype())?)?,
        };
        let indices_nulls_zeroed = indices_nulls_zeroed.execute::<PrimitiveArray>(ctx)?;
        let buffer = match_each_integer_ptype!(indices_nulls_zeroed.ptype(), |I| {
            take_valid_indices(
                array.bit_buffer_view(),
                indices_nulls_zeroed.as_slice::<I>(),
            )
        });

        Ok(Some(
            BoolArray::new(buffer, array.validity()?.take(indices)?).into_array(),
        ))
    }
}

fn take_piecewise_sequence(
    array: ArrayView<'_, Bool>,
    indices: ArrayView<'_, PiecewiseSequence>,
    indices_ref: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some((starts, lengths)) = execute_unit_multiplier_index_arrays(indices, ctx)? else {
        return Ok(None);
    };
    let buffer = match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
        match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
            take_piecewise_bits(
                &array.to_bit_buffer(),
                starts.as_slice::<S>(),
                lengths.as_slice::<L>(),
                indices_ref.len(),
            )?
        })
    });

    Ok(Some(
        BoolArray::new(buffer, array.validity()?.take(indices_ref)?).into_array(),
    ))
}

fn take_valid_indices<I: AsPrimitive<usize>>(bools: BitBufferView<'_>, indices: &[I]) -> BitBuffer {
    // For boolean arrays that roughly fit into a single page (at least, on Linux), it's worth
    // the overhead to convert to a Vec<bool>.
    if bools.len() <= 4096 {
        let bools = bools.iter().collect_vec();
        take_byte_bool(bools, indices)
    } else {
        take_bool_impl(bools, indices)
    }
}

fn take_byte_bool<I: AsPrimitive<usize>>(bools: Vec<bool>, indices: &[I]) -> BitBuffer {
    BitBuffer::collect_bool(indices.len(), |idx| {
        bools[unsafe { indices.get_unchecked(idx).as_() }]
    })
}

fn take_bool_impl<I: AsPrimitive<usize>>(bools: BitBufferView<'_>, indices: &[I]) -> BitBuffer {
    // We dereference to underlying buffer to avoid access cost on every index.
    let buffer = bools.inner();
    BitBuffer::collect_bool(indices.len(), |idx| {
        // SAFETY: we can take from the indices unchecked since collect_bool just iterates len.
        let idx = unsafe { indices.get_unchecked(idx).as_() };
        get_bit(buffer, bools.offset() + idx)
    })
}

fn take_piecewise_bits<S, L>(
    source: &BitBuffer,
    starts: &[S],
    lengths: &[L],
    output_len: usize,
) -> VortexResult<BitBuffer>
where
    S: UnsignedPType,
    L: UnsignedPType,
{
    validate_index_ranges(source.len(), starts, lengths, output_len)?;

    let mut values = BitBufferMut::with_capacity(output_len);
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = start.as_();
        let length = length.as_();
        values.append_buffer(&source.slice(start..start + length));
    }

    Ok(values.freeze())
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::IntoArray as _;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::bool::BoolArrayExt;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::validity::Validity;

    #[test]
    fn take_nullable() {
        let mut ctx = array_session().create_execution_ctx();
        let reference = BoolArray::from_iter(vec![
            Some(false),
            Some(true),
            Some(false),
            None,
            Some(false),
        ]);

        let b = reference
            .take(buffer![0, 3, 4].into_array())
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
        assert_eq!(
            b.to_bit_buffer(),
            BoolArray::from_iter([Some(false), None, Some(false)]).to_bit_buffer()
        );

        let all_invalid_indices = PrimitiveArray::from_option_iter([None::<i32>, None, None]);
        let b = reference.take(all_invalid_indices.into_array()).unwrap();
        assert_arrays_eq!(b, BoolArray::from_iter([None, None, None]), &mut ctx);
    }

    #[test]
    fn test_bool_array_take_with_null_out_of_bounds_indices() {
        let mut ctx = array_session().create_execution_ctx();
        let values = BoolArray::from_iter(vec![Some(false), Some(true), None, None, Some(false)]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([true, true, false]).into_array()),
        );
        let actual = values.take(indices.into_array()).unwrap();

        // position 3 is null, the third index is null
        assert_arrays_eq!(
            actual,
            BoolArray::from_iter([Some(false), None, None]),
            &mut ctx
        );
    }

    #[test]
    fn test_non_null_bool_array_take_with_null_out_of_bounds_indices() {
        let mut ctx = array_session().create_execution_ctx();
        let values = BoolArray::from_iter(vec![false, true, false, true, false]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([true, true, false]).into_array()),
        );
        let actual = values.take(indices.into_array()).unwrap();
        // the third index is null
        assert_arrays_eq!(
            actual,
            BoolArray::from_iter([Some(false), Some(true), None]),
            &mut ctx
        );
    }

    #[test]
    fn test_bool_array_take_all_null_indices() {
        let mut ctx = array_session().create_execution_ctx();
        let values = BoolArray::from_iter(vec![Some(false), Some(true), None, None, Some(false)]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([false, false, false]).into_array()),
        );
        let actual = values.take(indices.into_array()).unwrap();
        assert_arrays_eq!(actual, BoolArray::from_iter([None, None, None]), &mut ctx);
    }

    #[test]
    fn test_non_null_bool_array_take_all_null_indices() {
        let mut ctx = array_session().create_execution_ctx();
        let values = BoolArray::from_iter(vec![false, true, false, true, false]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([false, false, false]).into_array()),
        );
        let actual = values.take(indices.into_array()).unwrap();
        assert_arrays_eq!(actual, BoolArray::from_iter([None, None, None]), &mut ctx);
    }

    #[rstest]
    #[case(BoolArray::from_iter([true, false, true, true, false]))]
    #[case(BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]))]
    #[case(BoolArray::from_iter([true, false]))]
    #[case(BoolArray::from_iter([true]))]
    fn test_take_bool_conformance(#[case] array: BoolArray) {
        test_take_conformance(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::Columnar;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::PiecewiseSequence;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::piecewise_sequence::constant_unsigned_usize;
use crate::arrays::piecewise_sequence::copy_slice_to_spare;
use crate::arrays::piecewise_sequence::maybe_contiguous_slices;
use crate::dtype::IntegerPType;
use crate::dtype::NativeDecimalType;
use crate::dtype::UnsignedPType;
use crate::executor::ExecutionCtx;
use crate::match_each_decimal_value_type;
use crate::match_each_integer_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

impl TakeExecute for Decimal {
    fn take(
        array: ArrayView<'_, Decimal>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(piecewise_indices) = indices.as_opt::<PiecewiseSequence>()
            && let Some(taken) = take_contiguous_ranges(array, piecewise_indices, indices, ctx)?
        {
            return Ok(Some(taken));
        }

        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let validity = array.validity()?.take(&indices.clone().into_array())?;

        // TODO(joe): if the true count of take indices validity is low, only take array values with
        // valid indices.
        let decimal = match_each_decimal_value_type!(array.values_type(), |D| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                let buffer =
                    take_to_buffer::<I, D>(indices.as_slice::<I>(), array.buffer::<D>().as_slice());
                // SAFETY: Take operation preserves decimal dtype and creates valid buffer.
                // Validity is computed correctly from the parent array and indices.
                unsafe { DecimalArray::new_unchecked(buffer, array.decimal_dtype(), validity) }
            })
        });

        Ok(Some(decimal.into_array()))
    }
}

fn take_contiguous_ranges(
    array: ArrayView<'_, Decimal>,
    indices: ArrayView<'_, PiecewiseSequence>,
    indices_ref: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some((starts, lengths)) = maybe_contiguous_slices(indices, ctx)? else {
        return Ok(None);
    };
    let validity = array.validity()?.take(indices_ref)?;
    let output_len = indices_ref.len();
    let taken = match lengths {
        Columnar::Constant(lengths) => {
            let length = constant_unsigned_usize(&lengths);
            take_slices_constant_length(array, &starts, length, validity, output_len)?
        }
        Columnar::Canonical(lengths) => {
            let lengths = lengths.into_primitive();
            take_slices(array, &starts, &lengths, validity, output_len)?
        }
    };
    Ok(Some(taken))
}

fn take_slices_constant_length(
    array: ArrayView<'_, Decimal>,
    starts: &PrimitiveArray,
    length: usize,
    validity: Validity,
    output_len: usize,
) -> VortexResult<ArrayRef> {
    match_each_decimal_value_type!(array.values_type(), |D| {
        take_slices_constant_length_typed::<D>(array, starts, length, validity, output_len)
    })
}

fn take_slices_constant_length_typed<D>(
    array: ArrayView<'_, Decimal>,
    starts: &PrimitiveArray,
    length: usize,
    validity: Validity,
    output_len: usize,
) -> VortexResult<ArrayRef>
where
    D: NativeDecimalType,
{
    match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
        let values = take_slices_constant_length_to_buffer::<S, D>(
            starts.as_slice::<S>(),
            length,
            array.buffer::<D>().as_slice(),
            output_len,
        )?;

        // SAFETY: contiguous gather preserves the decimal dtype and value representation.
        Ok(
            unsafe { DecimalArray::new_unchecked(values, array.decimal_dtype(), validity) }
                .into_array(),
        )
    })
}

fn take_slices(
    array: ArrayView<'_, Decimal>,
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    validity: Validity,
    output_len: usize,
) -> VortexResult<ArrayRef> {
    match_each_decimal_value_type!(array.values_type(), |D| {
        take_slices_typed::<D>(array, starts, lengths, validity, output_len)
    })
}

fn take_slices_typed<D>(
    array: ArrayView<'_, Decimal>,
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    validity: Validity,
    output_len: usize,
) -> VortexResult<ArrayRef>
where
    D: NativeDecimalType,
{
    match_each_unsigned_integer_ptype!(starts.ptype(), |S| {
        take_slices_start_typed::<D, S>(array, starts, lengths, validity, output_len)
    })
}

fn take_slices_start_typed<D, S>(
    array: ArrayView<'_, Decimal>,
    starts: &PrimitiveArray,
    lengths: &PrimitiveArray,
    validity: Validity,
    output_len: usize,
) -> VortexResult<ArrayRef>
where
    D: NativeDecimalType,
    S: UnsignedPType,
{
    match_each_unsigned_integer_ptype!(lengths.ptype(), |L| {
        let values = take_slices_to_buffer::<S, L, D>(
            starts.as_slice::<S>(),
            lengths.as_slice::<L>(),
            array.buffer::<D>().as_slice(),
            output_len,
        )?;

        // SAFETY: contiguous gather preserves the decimal dtype and value representation.
        Ok(
            unsafe { DecimalArray::new_unchecked(values, array.decimal_dtype(), validity) }
                .into_array(),
        )
    })
}

fn take_to_buffer<I: IntegerPType, T: NativeDecimalType>(indices: &[I], values: &[T]) -> Buffer<T> {
    indices.iter().map(|idx| values[idx.as_()]).collect()
}

fn take_slices_constant_length_to_buffer<S, T>(
    starts: &[S],
    length: usize,
    values: &[T],
    output_len: usize,
) -> VortexResult<Buffer<T>>
where
    S: UnsignedPType,
    T: NativeDecimalType,
{
    let computed_len = starts
        .len()
        .checked_mul(length)
        .ok_or_else(|| vortex_err!("PiecewiseSequenceArray output length overflows usize"))?;
    vortex_ensure!(
        computed_len == output_len,
        "PiecewiseSequenceArray expanded length {computed_len} does not match declared length {output_len}"
    );

    let mut result = BufferMut::<T>::with_capacity(output_len);
    let mut cursor = 0usize;
    for &start in starts {
        let start = start.as_();
        cursor = copy_slice_to_spare(&mut result, cursor, &values[start..][..length], output_len)?;
    }

    vortex_ensure!(
        cursor == output_len,
        "PiecewiseSequenceArray expanded length {cursor} does not match declared length {output_len}"
    );
    // SAFETY: `copy_slice_to_spare` checked every write against `output_len`, and `cursor ==
    // output_len` proves all slots were initialized.
    unsafe { result.set_len(output_len) };
    Ok(result.freeze())
}

fn take_slices_to_buffer<S, L, T>(
    starts: &[S],
    lengths: &[L],
    values: &[T],
    output_len: usize,
) -> VortexResult<Buffer<T>>
where
    S: UnsignedPType,
    L: UnsignedPType,
    T: NativeDecimalType,
{
    let mut result = BufferMut::<T>::with_capacity(output_len);
    let mut cursor = 0usize;
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = start.as_();
        let length = length.as_();
        cursor = copy_slice_to_spare(&mut result, cursor, &values[start..][..length], output_len)?;
    }

    vortex_ensure!(
        cursor == output_len,
        "PiecewiseSequenceArray expanded length {} does not match declared length {output_len}",
        cursor
    );
    // SAFETY: `copy_slice_to_spare` checked every write against `output_len`, and `cursor ==
    // output_len` proves all slots were initialized.
    unsafe { result.set_len(output_len) };
    Ok(result.freeze())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::DecimalArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::DecimalDType;
    use crate::validity::Validity;

    #[test]
    fn test_take() {
        let mut ctx = array_session().create_execution_ctx();
        let ddtype = DecimalDType::new(19, 1);
        let array = DecimalArray::new(
            buffer![10i128, 11i128, 12i128, 13i128],
            ddtype,
            Validity::NonNullable,
        );

        let indices = buffer![0, 2, 3].into_array();
        let taken = array.take(indices).unwrap();

        let expected = DecimalArray::from_iter([10i128, 12, 13], ddtype);
        assert_arrays_eq!(expected, taken, &mut ctx);
    }

    #[test]
    fn test_take_null_indices() {
        let mut ctx = array_session().create_execution_ctx();
        let ddtype = DecimalDType::new(19, 1);
        let array = DecimalArray::new(
            buffer![i128::MAX, 11i128, 12i128, 13i128],
            ddtype,
            Validity::NonNullable,
        );

        let indices = PrimitiveArray::from_option_iter([None, Some(2), Some(3)]).into_array();
        let taken = array.take(indices).unwrap();

        let expected = DecimalArray::from_option_iter([None, Some(12i128), Some(13)], ddtype);
        assert_arrays_eq!(expected, taken, &mut ctx);
    }

    #[rstest]
    #[case(DecimalArray::new(
        buffer![100i128, 200i128, 300i128, 400i128, 500i128],
        DecimalDType::new(19, 2),
        Validity::NonNullable,
    ))]
    #[case(DecimalArray::new(
        buffer![10i64, 20i64, 30i64, 40i64, 50i64],
        DecimalDType::new(10, 1),
        Validity::NonNullable,
    ))]
    #[case(DecimalArray::new(
        buffer![1i32, 2i32, 3i32, 4i32, 5i32],
        DecimalDType::new(5, 0),
        Validity::NonNullable,
    ))]
    #[case(DecimalArray::new(
        buffer![1000i128, 2000i128, 3000i128, 4000i128, 5000i128],
        DecimalDType::new(19, 3),
        Validity::from_iter([true, false, true, true, false]),
    ))]
    #[case(DecimalArray::new(
        buffer![42i128],
        DecimalDType::new(19, 0),
        Validity::NonNullable,
    ))]
    #[case({
        let values: Vec<i128> = (0..100).map(|i| i * 1000).collect();
        DecimalArray::new(
            Buffer::from_iter(values),
            DecimalDType::new(19, 4),
            Validity::NonNullable,
        )
    })]
    fn test_take_decimal_conformance(#[case] array: DecimalArray) {
        test_take_conformance(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}

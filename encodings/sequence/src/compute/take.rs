// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::Sequence;
use crate::eval;
use crate::eval::SequenceValue;

fn take_inner<T: IntegerPType, O: SequenceValue>(
    base: O,
    multiplier: O,
    indices: &[T],
    indices_mask: Mask,
    result_nullability: Nullability,
    len: usize,
) -> ArrayRef {
    match indices_mask.bit_buffer() {
        AllOr::All => PrimitiveArray::new(
            Buffer::from_trusted_len_iter(indices.iter().map(|i| {
                if i.as_() >= len {
                    vortex_panic!(OutOfBounds: i.as_(), 0, len);
                }
                eval::wrapping_value(base, multiplier, i.as_())
            })),
            Validity::from(result_nullability),
        )
        .into_array(),
        AllOr::None => ConstantArray::new(
            Scalar::null(DType::Primitive(O::PTYPE, Nullability::Nullable)),
            indices.len(),
        )
        .into_array(),
        AllOr::Some(b) => {
            let buffer =
                Buffer::from_trusted_len_iter(indices.iter().enumerate().map(|(mask_index, i)| {
                    if b.value(mask_index) {
                        if i.as_() >= len {
                            vortex_panic!(OutOfBounds: i.as_(), 0, len);
                        }

                        eval::wrapping_value(base, multiplier, i.as_())
                    } else {
                        O::zero()
                    }
                }));
            PrimitiveArray::new(buffer, Validity::from(b.clone())).into_array()
        }
    }
}

fn take_with_typed_indices<T: IntegerPType>(
    array: ArrayView<'_, Sequence>,
    indices: &[T],
    indices_mask: Mask,
    result_nullability: Nullability,
) -> VortexResult<ArrayRef> {
    match_each_integer_ptype!(array.dtype().as_ptype(), |O| {
        let (base, multiplier) = array.wrapping_parts::<O>()?;
        Ok(take_inner::<T, O>(
            base,
            multiplier,
            indices,
            indices_mask,
            result_nullability,
            array.len(),
        ))
    })
}

fn take_sequence(
    array: ArrayView<'_, Sequence>,
    indices: &PrimitiveArray,
    indices_mask: Mask,
    result_nullability: Nullability,
) -> VortexResult<ArrayRef> {
    match_each_integer_ptype!(indices.ptype(), |T| {
        take_with_typed_indices::<T>(
            array,
            indices.as_slice::<T>(),
            indices_mask,
            result_nullability,
        )
    })
}

impl TakeExecute for Sequence {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let mask = indices.validity()?.execute_mask(indices.len(), ctx)?;
        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let result_nullability = array.dtype().nullability() | indices.dtype().nullability();

        take_sequence(array, &indices, mask, result_nullability).map(Some)
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::PValue;

    use crate::Sequence;
    use crate::SequenceArray;

    #[rstest]
    #[case::basic_sequence(Sequence::try_new_typed(
        0i32,
        1i32,
        Nullability::NonNullable,
        10
    ).unwrap())]
    #[case::sequence_with_multiplier(Sequence::try_new_typed(
        10i32,
        5i32,
        Nullability::Nullable,
        20
    ).unwrap())]
    #[case::sequence_i64(Sequence::try_new_typed(
        100i64,
        10i64,
        Nullability::NonNullable,
        50
    ).unwrap())]
    #[case::sequence_u32(Sequence::try_new_typed(
        0u32,
        2u32,
        Nullability::NonNullable,
        100
    ).unwrap())]
    #[case::sequence_negative_step(Sequence::try_new_typed(
        1000i32,
        -10i32,
        Nullability::Nullable,
        30
    ).unwrap())]
    #[case::sequence_constant(Sequence::try_new_typed(
        42i32,
        0i32,  // multiplier of 0 means all values are the same
        Nullability::Nullable,
        15
    ).unwrap())]
    #[case::sequence_i16(Sequence::try_new_typed(
        -100i16,
        3i16,
        Nullability::NonNullable,
        25
    ).unwrap())]
    #[case::sequence_large(Sequence::try_new_typed(
        0i64,
        1i64,
        Nullability::Nullable,
        1000
    ).unwrap())]
    #[case::sequence_descending_u8(Sequence::try_new(
        PValue::from(200i32),
        PValue::from(-3i32),
        PType::U8,
        Nullability::NonNullable,
        60
    ).unwrap())]
    #[case::sequence_past_i64_max(Sequence::try_new(
        PValue::from(0i64),
        PValue::from(1i64 << 62),
        PType::U64,
        Nullability::NonNullable,
        4
    ).unwrap())]
    #[case::sequence_constant_longer_than_u8(Sequence::try_new(
        PValue::from(7u8),
        PValue::from(0i32),
        PType::U8,
        Nullability::NonNullable,
        500
    ).unwrap())]
    fn sequence_take_conformance(#[case] sequence: SequenceArray) {
        test_take_conformance(
            &sequence.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_bounds_check() {
        let array = Sequence::try_new_typed(0i32, 1i32, Nullability::NonNullable, 10).unwrap();
        let indices = PrimitiveArray::from_iter([0i32, 20]);
        let _array = array
            .take(indices.into_array())
            .unwrap()
            .execute::<Canonical>(&mut array_session().create_execution_ctx())
            .unwrap();
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::UnsignedPType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::Sequence;
use crate::eval;
use crate::eval::SequenceValue;

/// Evaluates the sequence at every index. Out-of-bounds indices are reported through the lane
/// kernel so the hot loop stays branch-free, and null indices are exempt from the bounds check.
///
/// Both `T` and `O` are unsigned: indices are non-negative, and wrapping arithmetic produces the
/// same bits in either signedness, so the result is reinterpreted back to the sequence ptype. This
/// keeps the kernel to 4x4 instantiations instead of 8x8.
fn take_inner<T: UnsignedPType, O: SequenceValue + UnsignedPType>(
    array: ArrayView<'_, Sequence>,
    indices: &[T],
    indices_mask: Mask,
    result_nullability: Nullability,
    allocator: BufferAllocatorRef,
) -> VortexResult<ArrayRef> {
    let (base, multiplier) = array.wrapping_parts::<O>()?;
    let ptype = array.dtype().as_ptype();
    let len = array.len();
    let value_at = |index: T| {
        let index: usize = index.as_();
        (index < len).then(|| eval::wrapping_value(base, multiplier, index))
    };

    let mut buffer = BufferMut::<O>::with_capacity_in(indices.len(), allocator);
    let out = &mut buffer.spare_capacity_mut()[..indices.len()];
    let validity = match indices_mask.bit_buffer() {
        AllOr::All => {
            if let Err(position) = indices.try_map_into(out, value_at) {
                vortex_panic!(
                    "index {} out of bounds from {} to {}",
                    indices[position].as_(),
                    0,
                    len
                );
            }
            Validity::from(result_nullability)
        }
        AllOr::None => {
            return Ok(ConstantArray::new(
                Scalar::null(DType::Primitive(ptype, Nullability::Nullable)),
                indices.len(),
            )
            .into_array());
        }
        AllOr::Some(bits) => {
            if let Err(position) = indices.try_map_masked_into(bits, out, value_at) {
                vortex_panic!(
                    "index {} out of bounds from {} to {}",
                    indices[position].as_(),
                    0,
                    len
                );
            }
            Validity::from(bits.clone())
        }
    };
    // SAFETY: the lane kernel wrote every lane before returning `Ok`.
    unsafe { buffer.set_len(indices.len()) };

    Ok(PrimitiveArray::new(buffer.freeze(), validity)
        .reinterpret_cast(ptype)
        .into_array())
}

fn take_sequence(
    array: ArrayView<'_, Sequence>,
    indices: &PrimitiveArray,
    indices_mask: Mask,
    result_nullability: Nullability,
    allocator: BufferAllocatorRef,
) -> VortexResult<ArrayRef> {
    let indices = indices.reinterpret_cast(indices.ptype().to_unsigned());
    match_each_unsigned_integer_ptype!(indices.ptype(), |T| {
        match_each_unsigned_integer_ptype!(array.dtype().as_ptype().to_unsigned(), |O| {
            take_inner::<T, O>(
                array,
                indices.as_slice::<T>(),
                indices_mask,
                result_nullability,
                allocator,
            )
        })
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

        take_sequence(
            array,
            &indices,
            mask,
            result_nullability,
            ctx.allocator().clone(),
        )
        .map(Some)
    }
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use hegel::TestCase;
    use hegel::generators as gs;
    use num_traits::Bounded;
    use num_traits::ToPrimitive;
    use rstest::rstest;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::match_each_integer_ptype;
    use vortex_array::scalar::PValue;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Sequence;
    use crate::SequenceArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

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
        test_take_conformance(&sequence.into_array(), &mut SESSION.create_execution_ctx());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_bounds_check() {
        let array = Sequence::try_new_typed(0i32, 1i32, Nullability::NonNullable, 10).unwrap();
        let indices = PrimitiveArray::from_iter([0i32, 20]);
        let _array = array
            .take(indices.into_array())
            .unwrap()
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())
            .unwrap();
    }

    #[rstest]
    #[case::i8(PrimitiveArray::from_iter([0i8, -1]))]
    #[case::i16(PrimitiveArray::from_iter([0i16, -1]))]
    #[case::i32(PrimitiveArray::from_iter([0i32, -1]))]
    #[case::i64(PrimitiveArray::from_iter([0i64, -1]))]
    #[should_panic(expected = "out of bounds")]
    fn test_negative_index_out_of_bounds(#[case] indices: PrimitiveArray) {
        let array = Sequence::try_new_typed(0i32, 1i32, Nullability::NonNullable, 10).unwrap();
        let _array = array
            .take(indices.into_array())
            .unwrap()
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())
            .unwrap();
    }

    #[test]
    fn test_null_negative_index_ignored() -> VortexResult<()> {
        let array = Sequence::try_new_typed(-5i32, 2i32, Nullability::NonNullable, 10)?;
        let indices = PrimitiveArray::new(
            buffer![3i32, -1, 0],
            Validity::from_iter([true, false, true]),
        );
        let taken = array.take(indices.into_array())?;
        assert_arrays_eq!(
            taken,
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(-5)]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    const PTYPES: [PType; 8] = [
        PType::I8,
        PType::I16,
        PType::I32,
        PType::I64,
        PType::U8,
        PType::U16,
        PType::U32,
        PType::U64,
    ];

    /// Rows per generated sequence and per generated set of indices.
    const MAX_LEN: usize = 64;

    fn ptype_bounds(ptype: PType) -> (i128, i128) {
        match_each_integer_ptype!(ptype, |P| {
            (
                <P as Bounded>::min_value()
                    .to_i128()
                    .vortex_expect("integer min fits i128"),
                <P as Bounded>::max_value()
                    .to_i128()
                    .vortex_expect("integer max fits i128"),
            )
        })
    }

    /// Every value drawn here lies in `[i64::MIN, u64::MAX]`.
    fn pvalue(value: i128) -> PValue {
        match i64::try_from(value) {
            Ok(value) => PValue::I64(value),
            Err(_) => PValue::U64(u64::try_from(value).vortex_expect("value fits u64")),
        }
    }

    /// Taking from ascending and descending sequences of every integer type, with negative and
    /// positive bases, matches an exact `i128` evaluation of `base + index * multiplier`. Null
    /// indices hold negative (signed) or out-of-bounds (unsigned) values that must be ignored.
    #[hegel::test]
    fn take_matches_exact_model(tc: TestCase) {
        check_take_matches_exact_model(&tc).vortex_expect("sequence take");
    }

    fn check_take_matches_exact_model(tc: &TestCase) -> VortexResult<()> {
        let ptype = tc.draw(gs::sampled_from(&PTYPES));
        let len = tc.draw(gs::integers::<usize>().min_value(1).max_value(MAX_LEN));
        let (min, max) = ptype_bounds(ptype);

        // Bound the multiplier so some base keeps every value in range, while still reaching
        // both signs and the extremes of the type.
        let steps = i128::try_from(len - 1)?;
        let span = if steps == 0 {
            i128::from(i64::MAX)
        } else {
            (max - min) / steps
        };
        let multiplier = tc.draw(
            gs::integers::<i128>()
                .min_value(-span.min(i128::from(i64::MAX)))
                .max_value(span),
        );
        let last_offset = multiplier * steps;
        let base = tc.draw(
            gs::integers::<i128>()
                .min_value(min.max(min - last_offset))
                .max_value(max.min(max - last_offset)),
        );
        let nullability = if tc.draw(gs::booleans()) {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        };
        let sequence =
            Sequence::try_new(pvalue(base), pvalue(multiplier), ptype, nullability, len)?;

        let index_ptype = tc.draw(gs::sampled_from(&PTYPES));
        let nullable_indices = tc.draw(gs::booleans());
        let index_count = tc.draw(gs::integers::<usize>().max_value(MAX_LEN));
        let mut indices = Vec::with_capacity(index_count);
        let mut valid = Vec::with_capacity(index_count);
        for _ in 0..index_count {
            let is_valid = !nullable_indices || tc.draw(gs::booleans());
            let index = if is_valid {
                tc.draw(
                    gs::integers::<i64>()
                        .min_value(0)
                        .max_value(i64::try_from(len)? - 1),
                )
            } else if index_ptype.is_signed_int() {
                tc.draw(gs::integers::<i64>().min_value(-100).max_value(-1))
            } else {
                tc.draw(
                    gs::integers::<i64>()
                        .min_value(i64::try_from(len)?)
                        .max_value(200),
                )
            };
            indices.push(index);
            valid.push(is_valid);
        }

        let indices_validity = if nullable_indices {
            Validity::from_iter(valid.iter().copied())
        } else {
            Validity::NonNullable
        };
        let indices_array = match_each_integer_ptype!(index_ptype, |I| {
            PrimitiveArray::new(
                Buffer::<I>::from_iter(
                    indices
                        .iter()
                        .map(|&index| num_traits::cast(index).vortex_expect("index fits")),
                ),
                indices_validity.clone(),
            )
        });

        let expected_values = indices.iter().zip(&valid).map(|(&index, &is_valid)| {
            if is_valid {
                base + multiplier * i128::from(index)
            } else {
                base
            }
        });
        let expected_validity = if nullable_indices {
            indices_validity
        } else {
            Validity::from(nullability)
        };
        let expected = match_each_integer_ptype!(ptype, |P| {
            PrimitiveArray::new(
                Buffer::<P>::from_iter(
                    expected_values.map(|value| num_traits::cast(value).vortex_expect("in range")),
                ),
                expected_validity,
            )
        });

        let taken = sequence.into_array().take(indices_array.into_array())?;
        assert_arrays_eq!(taken, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }
}

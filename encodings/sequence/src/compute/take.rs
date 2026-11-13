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

/// Evaluates the sequence at every index.
///
/// `T` and `O` are unsigned to keep 4x4 instantiations, because wrapping arithmetic produces the
/// same bits in either signedness.
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
    use rstest::rstest;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::dtype::NativePType;
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

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_negative_index_out_of_bounds() {
        let array = Sequence::try_new_typed(0i32, 1i32, Nullability::NonNullable, 10).unwrap();
        let indices = PrimitiveArray::from_iter([0i32, -1]);
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

    const INDEX_PTYPES: [PType; 8] = [
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

    /// Taking from a generated sequence matches taking from its canonical values.
    #[hegel::test]
    fn take_matches_canonical(tc: TestCase) {
        match tc.draw(gs::integers::<u8>().max_value(7)) {
            0 => check_take::<u8>(&tc),
            1 => check_take::<u16>(&tc),
            2 => check_take::<u32>(&tc),
            3 => check_take::<u64>(&tc),
            4 => check_take::<i8>(&tc),
            5 => check_take::<i16>(&tc),
            6 => check_take::<i32>(&tc),
            _ => check_take::<i64>(&tc),
        }
        .vortex_expect("sequence take");
    }

    fn check_take<P>(tc: &TestCase) -> VortexResult<()>
    where
        P: NativePType + gs::Integer + Into<PValue> + 'static,
    {
        let mut ctx = SESSION.create_execution_ctx();
        let base: P = tc.draw(gs::integers());
        let multiplier: i64 = tc.draw(gs::integers());
        let len: usize = tc.draw(gs::integers().min_value(1).max_value(MAX_LEN));
        let Ok(sequence) = Sequence::try_new(
            base.into(),
            PValue::from(multiplier),
            P::PTYPE,
            Nullability::NonNullable,
            len,
        ) else {
            tc.reject()
        };
        let sequence = sequence.into_array();

        let indices: Vec<u8> = tc.draw(gs::vecs(
            gs::integers::<u8>().max_value(u8::try_from(len - 1)?),
        ));
        let index_ptype = tc.draw(gs::sampled_from(&INDEX_PTYPES));
        let indices = match_each_integer_ptype!(index_ptype, |I| {
            Buffer::<I>::from_iter(
                indices
                    .iter()
                    .map(|&index| num_traits::cast::<u8, I>(index).vortex_expect("index fits")),
            )
            .into_array()
        });

        let expected = sequence
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array()
            .take(indices.clone())?;
        let taken = sequence.take(indices)?;
        assert_arrays_eq!(taken, expected, &mut ctx);
        Ok(())
    }
}

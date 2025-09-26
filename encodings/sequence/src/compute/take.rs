// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::cast::NumCast;
use vortex_array::arrays::{ConstantArray, PrimitiveArray};
use vortex_array::compute::{TakeKernel, TakeKernelAdapter};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};
use vortex_buffer::Buffer;
use vortex_dtype::{
    DType, IntegerPType, NativePType, Nullability, match_each_integer_ptype,
    match_each_native_ptype,
};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{AllOr, Mask};
use vortex_scalar::Scalar;

use crate::{SequenceArray, SequenceVTable};

impl TakeKernel for SequenceVTable {
    fn take(&self, array: &SequenceArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let mask = indices.validity_mask();
        let indices = indices.to_primitive();
        let result_nullability = array.dtype().nullability() | indices.dtype().nullability();

        Ok(match_each_integer_ptype!(indices.ptype(), |T| {
            let indices = indices.as_slice::<T>();
            match_each_native_ptype!(array.ptype(), |S| {
                let mul = array.multiplier().as_primitive::<S>();
                let base = array.base().as_primitive::<S>();
                take(mul, base, indices, mask, result_nullability)
            })
        }))
    }
}

fn take<T: IntegerPType, S: NativePType>(
    mul: S,
    base: S,
    indices: &[T],
    indices_mask: Mask,
    result_nullability: Nullability,
) -> ArrayRef {
    match indices_mask.boolean_buffer() {
        AllOr::All => PrimitiveArray::new(
            Buffer::from_trusted_len_iter(indices.iter().map(|i| {
                let i = <S as NumCast>::from::<T>(*i).vortex_expect("all indices fit");
                base + i * mul
            })),
            Validity::from(result_nullability),
        )
        .into_array(),
        AllOr::None => ConstantArray::new(
            Scalar::null(DType::Primitive(S::PTYPE, Nullability::Nullable)),
            indices.len(),
        )
        .into_array(),
        AllOr::Some(b) => {
            let buffer =
                Buffer::from_trusted_len_iter(indices.iter().enumerate().map(|(mask_index, i)| {
                    if b.value(mask_index) {
                        let i =
                            <S as NumCast>::from::<T>(*i).vortex_expect("all valid indices fit");
                        base + i * mul
                    } else {
                        S::zero()
                    }
                }));
            PrimitiveArray::new(buffer, Validity::from(b.clone())).into_array()
        }
    }
}

register_kernel!(TakeKernelAdapter(SequenceVTable).lift());

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_dtype::Nullability;

    use crate::SequenceArray;

    #[rstest]
    #[case::basic_sequence(SequenceArray::typed_new(
        0i32,
        1i32,
        Nullability::NonNullable,
        10
    ).unwrap())]
    #[case::sequence_with_multiplier(SequenceArray::typed_new(
        10i32,
        5i32,
        Nullability::Nullable,
        20
    ).unwrap())]
    #[case::sequence_i64(SequenceArray::typed_new(
        100i64,
        10i64,
        Nullability::NonNullable,
        50
    ).unwrap())]
    #[case::sequence_u32(SequenceArray::typed_new(
        0u32,
        2u32,
        Nullability::NonNullable,
        100
    ).unwrap())]
    #[case::sequence_negative_step(SequenceArray::typed_new(
        1000i32,
        -10i32,
        Nullability::Nullable,
        30
    ).unwrap())]
    #[case::sequence_constant(SequenceArray::typed_new(
        42i32,
        0i32,  // multiplier of 0 means all values are the same
        Nullability::Nullable,
        15
    ).unwrap())]
    #[case::sequence_i16(SequenceArray::typed_new(
        -100i16,
        3i16,
        Nullability::NonNullable,
        25
    ).unwrap())]
    #[case::sequence_large(SequenceArray::typed_new(
        0i64,
        1i64,
        Nullability::Nullable,
        1000
    ).unwrap())]
    fn test_take_conformance(#[case] sequence: SequenceArray) {
        use vortex_array::compute::conformance::take::test_take_conformance;
        test_take_conformance(sequence.as_ref());
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::MaskedArray;
use crate::compute::TakeKernel;
use crate::compute::TakeKernelAdapter;
use crate::register_kernel;
use crate::validity::Validity;

impl TakeKernel for ConstantVTable {
    fn take(&self, array: &ConstantArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        match indices.validity_mask().bit_buffer() {
            AllOr::All => {
                let scalar = Scalar::new(
                    array
                        .scalar()
                        .dtype()
                        .union_nullability(indices.dtype().nullability()),
                    array.scalar().value().clone(),
                );
                Ok(ConstantArray::new(scalar, indices.len()).into_array())
            }
            AllOr::None => Ok(ConstantArray::new(
                Scalar::null(
                    array
                        .dtype()
                        .union_nullability(indices.dtype().nullability()),
                ),
                indices.len(),
            )
            .into_array()),
            AllOr::Some(v) => {
                let arr = ConstantArray::new(array.scalar().clone(), indices.len()).into_array();

                if array.scalar().is_null() {
                    return Ok(arr);
                }

                Ok(MaskedArray::try_new(arr, Validity::from(v.clone()))?.into_array())
            }
        }
    }
}

register_kernel!(TakeKernelAdapter(ConstantVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_mask::AllOr;
    use vortex_scalar::Scalar;

    use crate::Array;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::compute::take;
    use crate::validity::Validity;

    #[test]
    fn take_nullable_indices() {
        let array = ConstantArray::new(42, 10).to_array();
        let taken = take(
            &array,
            &PrimitiveArray::new(
                buffer![0, 5, 7],
                Validity::from_iter(vec![false, true, false]),
            )
            .into_array(),
        )
        .unwrap();
        let valid_indices: &[usize] = &[1usize];
        assert_eq!(
            &array.dtype().with_nullability(Nullability::Nullable),
            taken.dtype()
        );
        assert_eq!(taken.to_primitive().as_slice::<i32>(), &[42, 42, 42]);
        assert_eq!(taken.validity_mask().indices(), AllOr::Some(valid_indices));
    }

    #[test]
    fn take_all_valid_indices() {
        let array = ConstantArray::new(42, 10).to_array();
        let taken = take(
            &array,
            &PrimitiveArray::new(buffer![0, 5, 7], Validity::AllValid).into_array(),
        )
        .unwrap();
        assert_eq!(
            &array.dtype().with_nullability(Nullability::Nullable),
            taken.dtype()
        );
        assert_eq!(taken.to_primitive().as_slice::<i32>(), &[42, 42, 42]);
        assert_eq!(taken.validity_mask().indices(), AllOr::All);
    }

    #[rstest]
    #[case(ConstantArray::new(42i32, 5))]
    #[case(ConstantArray::new(std::f64::consts::PI, 10))]
    #[case(ConstantArray::new(Scalar::from("hello"), 3))]
    #[case(ConstantArray::new(Scalar::null_typed::<i64>(), 5))]
    #[case(ConstantArray::new(true, 1))]
    fn test_take_constant_conformance(#[case] array: ConstantArray) {
        test_take_conformance(array.as_ref());
    }
}

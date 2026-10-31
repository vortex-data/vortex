// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::MaskedArray;
use crate::arrays::dict::TakeReduce;
use crate::scalar::Scalar;
use crate::validity::Validity;

impl TakeReduce for Constant {
    fn take(array: ArrayView<'_, Constant>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let result = match indices.validity()? {
            Validity::NonNullable | Validity::AllValid => {
                let scalar = Scalar::try_new(
                    array
                        .scalar()
                        .dtype()
                        .union_nullability(indices.dtype().nullability()),
                    array.scalar().value().cloned(),
                )?;
                ConstantArray::new(scalar, indices.len()).into_array()
            }
            Validity::AllInvalid => ConstantArray::new(
                Scalar::null(
                    array
                        .dtype()
                        .union_nullability(indices.dtype().nullability()),
                ),
                indices.len(),
            )
            .into_array(),
            validity @ Validity::Array(_) => {
                let arr = ConstantArray::new(array.scalar().clone(), indices.len()).into_array();

                if array.scalar().is_null() {
                    return Ok(Some(arr));
                }

                // SAFETY: `arr` repeats a non-null scalar, and the validity comes from the indices,
                // which have the same length as `arr`.
                unsafe { MaskedArray::new_unchecked(arr, validity) }.into_array()
            }
        };
        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use std::f64;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_mask::AllOr;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn take_nullable_indices() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = ConstantArray::new(42, 10).into_array();
        let taken = array
            .take(
                PrimitiveArray::new(
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
        assert_arrays_eq!(
            taken,
            PrimitiveArray::new(
                buffer![42i32, 42, 42],
                Validity::from_iter([false, true, false])
            ),
            &mut ctx
        );
        assert_eq!(
            taken
                .validity()
                .unwrap()
                .execute_mask(taken.len(), &mut ctx)
                .unwrap()
                .indices(),
            AllOr::Some(valid_indices)
        );
    }

    #[test]
    fn take_all_valid_indices() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = ConstantArray::new(42, 10).into_array();
        let taken = array
            .take(PrimitiveArray::new(buffer![0, 5, 7], Validity::AllValid).into_array())
            .unwrap();
        assert_eq!(
            &array.dtype().with_nullability(Nullability::Nullable),
            taken.dtype()
        );
        assert_arrays_eq!(
            taken,
            PrimitiveArray::new(buffer![42i32, 42, 42], Validity::AllValid),
            &mut ctx
        );
        assert_eq!(
            taken
                .validity()
                .unwrap()
                .execute_mask(taken.len(), &mut ctx)
                .unwrap()
                .indices(),
            AllOr::All
        );
    }

    #[rstest]
    #[case(ConstantArray::new(42i32, 5))]
    #[case(ConstantArray::new(f64::consts::PI, 10))]
    #[case(ConstantArray::new(Scalar::from("hello"), 3))]
    #[case(ConstantArray::new(Scalar::null_native::<i64>(), 5))]
    #[case(ConstantArray::new(true, 1))]
    fn test_take_constant_conformance(#[case] array: ConstantArray) {
        test_take_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}

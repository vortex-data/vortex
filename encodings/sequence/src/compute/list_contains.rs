// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementReduce;
use vortex_error::VortexResult;

use crate::array::Sequence;
use crate::compute::compare::Intersection;
use crate::compute::compare::find_intersection;

impl ListContainsElementReduce for Sequence {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(list_scalar) = list.as_constant() else {
            return Ok(None);
        };
        let DType::List(member_dtype, _) = list.dtype() else {
            return Ok(None);
        };
        if !member_dtype.eq_ignore_nullability(element.dtype()) {
            return Ok(None);
        }

        let Some(list_elements) = list_scalar.as_list().elements() else {
            return Ok(None);
        };
        if list_elements.is_empty() {
            return Ok(None);
        }

        let nullability = list.dtype().nullability() | element.dtype().nullability();

        let mut set_indices: Vec<usize> = Vec::new();
        for intercept in list_elements.iter() {
            let Some(intercept) = intercept.as_primitive().pvalue() else {
                continue;
            };
            match find_intersection(
                element.base(),
                element.multiplier(),
                element.len(),
                intercept,
            ) {
                // Non-integer elements do not match the sequence.
                None | Some(Intersection::None) => {}
                Some(Intersection::At(idx)) => set_indices.push(idx),
                Some(Intersection::All) => {
                    return Ok(Some(
                        ConstantArray::new(Scalar::bool(true, nullability), element.len())
                            .into_array(),
                    ));
                }
            }
        }

        Ok(Some(
            BoolArray::from_indices(element.len(), set_indices, nullability.into()).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Constant;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType::I32;
    use vortex_array::expr::list_contains;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;
    use vortex_session::VortexSession;

    use crate::Sequence;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn test_list_contains_seq() {
        let list_scalar = Scalar::list(
            Arc::new(I32.into()),
            vec![1.into(), 3.into()],
            Nullability::Nullable,
        );

        {
            // [1, 3] in  1
            //            2
            //            3
            let array = Sequence::try_new_typed(1, 1, Nullability::NonNullable, 3)
                .unwrap()
                .into_array();

            let expr = list_contains(lit(list_scalar.clone()), root());
            let result = array.into_array().apply(&expr).unwrap();
            let expected = BoolArray::from_iter([Some(true), Some(false), Some(true)]);
            assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        }

        {
            // [1, 3] in  1
            //            3
            //            5
            let array = Sequence::try_new_typed(1, 2, Nullability::NonNullable, 3)
                .unwrap()
                .into_array();

            let expr = list_contains(lit(list_scalar), root());
            let result = array.into_array().apply(&expr).unwrap();
            let expected = BoolArray::from_iter([Some(true), Some(true), Some(false)]);
            assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        }
    }

    #[test]
    fn test_list_contains_constant_sequence() {
        let list_scalar = Scalar::list(
            Arc::new(I32.into()),
            vec![7.into(), 42.into()],
            Nullability::Nullable,
        );

        let array = Sequence::try_new_typed(42i32, 0, Nullability::NonNullable, 3)
            .unwrap()
            .into_array();

        let expr = list_contains(lit(list_scalar), root());
        let result = array.apply(&expr).unwrap();
        let expected = BoolArray::from_iter([Some(true), Some(true), Some(true)]);
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
    }

    #[rstest]
    #[case::null_list(
        Scalar::null(DType::List(Arc::new(I32.into()), Nullability::Nullable)),
        [None, None, None]
    )]
    #[case::empty_list(
        Scalar::list(Arc::new(I32.into()), vec![], Nullability::Nullable),
        [Some(false), Some(false), Some(false)]
    )]
    fn test_constant_list_semantics(
        #[case] list_scalar: Scalar,
        #[case] expected: [Option<bool>; 3],
    ) {
        let array = Sequence::try_new_typed(1i32, 1, Nullability::NonNullable, 3)
            .unwrap()
            .into_array();
        let expr = list_contains(lit(list_scalar), root());

        let result = array.apply(&expr).unwrap();

        assert!(result.is::<Constant>());
        assert_arrays_eq!(
            result,
            BoolArray::from_iter(expected),
            &mut SESSION.create_execution_ctx()
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementReduce;
use vortex_array::scalar_fn::fns::list_contains::ListContainsOptions;
use vortex_error::VortexResult;

use crate::array::Sequence;
use crate::compute::compare::Intersection;
use crate::compute::compare::find_intersection;

impl ListContainsElementReduce for Sequence {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        options: &ListContainsOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(list_scalar) = list.as_constant() else {
            return Ok(None);
        };

        // A null list scalar has no elements to intersect with. Nothing checks this before the
        // reduce rule runs, so fall back to the generic implementation, which resolves a null
        // haystack to all-null rather than panicking here.
        let Some(list_elements) = list_scalar.as_list().elements() else {
            return Ok(None);
        };

        // The intersection search treats a null element as matching nothing, which under SQL null
        // semantics is only half the answer: there a non-match is unknown, which the search cannot
        // express.
        if options.sql_null_semantics && list_elements.iter().any(Scalar::is_null) {
            return Ok(None);
        }

        let nullability = options.result_nullability(list.dtype(), element.dtype());

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

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType::I32;
    use vortex_array::expr::list_contains;
    use vortex_array::expr::list_contains_opts;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::list_contains::ListContainsOptions;
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
    fn test_list_contains_null_list() {
        // A null haystack resolves to null for every row. The reduce rule used to assume the list
        // scalar was non-null and panicked instead of declining the reduction.
        let array = Sequence::try_new_typed(1, 1, Nullability::NonNullable, 3)
            .unwrap()
            .into_array();

        let null_list = Scalar::null(DType::List(Arc::new(I32.into()), Nullability::Nullable));
        let expr = list_contains(lit(null_list), root());
        let result = array.apply(&expr).unwrap();
        let expected = BoolArray::from_iter([None::<bool>, None, None]);
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_list_contains_null_element_semantics() {
        // The sequence kernel skips a null element, which is right by default. Under SQL null
        // semantics a non-match must be null instead, so a constant list holding a null goes to
        // the generic path instead.
        let element = DType::Primitive(I32, Nullability::Nullable);
        let set = Scalar::list(
            Arc::new(element.clone()),
            vec![
                Scalar::primitive(1i32, Nullability::Nullable),
                Scalar::null(element),
            ],
            Nullability::NonNullable,
        );
        let array = Sequence::try_new_typed(1, 1, Nullability::NonNullable, 3)
            .unwrap()
            .into_array();

        let result = array
            .clone()
            .apply(&list_contains(lit(set.clone()), root()))
            .unwrap();
        let expected = BoolArray::from_iter([true, false, false]);
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        let sql = ListContainsOptions {
            sql_null_semantics: true,
        };
        let result = array
            .apply(&list_contains_opts(lit(set), root(), sql))
            .unwrap();
        let expected = BoolArray::from_iter([Some(true), None, None]);
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
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
}

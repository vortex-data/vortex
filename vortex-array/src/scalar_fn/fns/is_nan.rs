// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use crate::dtype::DType;
use crate::match_each_float_ptype;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::RowVisitor;

/// Expression that checks for NaN values.
///
/// The function is strict: null inputs produce null outputs. Only primitive float inputs
/// are supported.
#[derive(Clone)]
pub struct IsNan;

impl RowFn for IsNan {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["input"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.is_nan");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        let ptype = match args {
            [DType::Primitive(ptype, _)] if ptype.is_float() => *ptype,
            _ => vortex_bail!("is_nan expects a single float input, got {args:?}"),
        };
        match_each_float_ptype!(ptype, |T| {
            visitor.visit_bool::<(T,), false>(|(value,)| value.is_nan())
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect as _;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::get_item;
    use crate::expr::is_nan;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn dtype() {
        assert_eq!(
            is_nan(root())
                .return_dtype(&DType::Primitive(PType::F32, Nullability::Nullable))
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );
        assert_eq!(
            is_nan(root())
                .return_dtype(&DType::Primitive(PType::F64, Nullability::NonNullable))
                .unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn dtype_rejects_non_float() {
        assert!(
            is_nan(root())
                .return_dtype(&DType::Primitive(PType::I32, Nullability::NonNullable))
                .is_err()
        );
        assert!(
            is_nan(root())
                .return_dtype(&DType::Bool(Nullability::NonNullable))
                .is_err()
        );
    }

    #[test]
    fn replace_children() {
        let expr = is_nan(root());
        expr.with_children([root()])
            .vortex_expect("operation should succeed in test");
    }

    #[test]
    fn evaluate_floats() {
        let test_array = PrimitiveArray::from_option_iter([
            Some(1.0f32),
            Some(f32::NAN),
            None,
            Some(f32::NEG_INFINITY),
            Some(-f32::NAN),
        ])
        .into_array();
        let expected = [Some(false), Some(true), None, Some(false), Some(true)];

        let result = test_array.clone().apply(&is_nan(root())).unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(result.dtype(), &DType::Bool(Nullability::Nullable));
        for (i, expected_value) in expected.iter().enumerate() {
            let expected_scalar = match expected_value {
                Some(value) => Scalar::bool(*value, Nullability::Nullable),
                None => Scalar::null(DType::Bool(Nullability::Nullable)),
            };
            assert_eq!(
                result
                    .execute_scalar(i, &mut array_session().create_execution_ctx())
                    .unwrap(),
                expected_scalar
            );
        }
    }

    #[test]
    fn evaluate_all_valid_floats() {
        let test_array = PrimitiveArray::new(
            buffer![1.0f64, f64::NAN, f64::INFINITY, -f64::NAN],
            Validity::NonNullable,
        )
        .into_array();
        let expected = [false, true, false, true];

        let result = test_array.clone().apply(&is_nan(root())).unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));
        for (i, expected_value) in expected.iter().enumerate() {
            assert_eq!(
                result
                    .execute_scalar(i, &mut array_session().create_execution_ctx())
                    .unwrap(),
                Scalar::bool(*expected_value, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn evaluate_all_null_floats() {
        let test_array = PrimitiveArray::from_option_iter([None::<f32>, None, None]).into_array();

        let result = test_array.clone().apply(&is_nan(root())).unwrap();

        assert_eq!(result.len(), test_array.len());
        for i in 0..result.len() {
            assert_eq!(
                result
                    .execute_scalar(i, &mut array_session().create_execution_ctx())
                    .unwrap(),
                Scalar::null(DType::Bool(Nullability::Nullable))
            );
        }
    }

    #[test]
    fn evaluate_constant() {
        let test_array = buffer![1.0f32, 2.0, 3.0].into_array();
        let cases = [
            (lit(f32::NAN), Some(true)),
            (lit(1.0f32), Some(false)),
            (
                lit(Scalar::null(DType::Primitive(
                    PType::F32,
                    Nullability::Nullable,
                ))),
                None,
            ),
        ];

        for (expr_child, expected_value) in cases {
            let result = test_array.clone().apply(&is_nan(expr_child)).unwrap();
            for i in 0..result.len() {
                let expected_scalar = match expected_value {
                    Some(value) => Scalar::bool(value, Nullability::Nullable),
                    None => Scalar::null(DType::Bool(Nullability::Nullable)),
                };
                assert_eq!(
                    result
                        .execute_scalar(i, &mut array_session().create_execution_ctx())
                        .unwrap(),
                    expected_scalar
                );
            }
        }
    }

    #[test]
    fn evaluate_rejects_non_float() {
        let test_array = buffer![1i32, 2, 3].into_array();
        assert!(test_array.apply(&is_nan(root())).is_err());
    }

    #[test]
    fn evaluate_sliced() {
        let test_array = buffer![1.0f32, f32::NAN, 2.0, f32::NAN, 3.0]
            .into_array()
            .slice(1..4)
            .unwrap();
        let expected = [true, false, true];

        let result = test_array.clone().apply(&is_nan(root())).unwrap();

        assert_eq!(result.len(), test_array.len());
        for (i, expected_value) in expected.iter().enumerate() {
            assert_eq!(
                result
                    .execute_scalar(i, &mut array_session().create_execution_ctx())
                    .unwrap(),
                Scalar::bool(*expected_value, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn test_display() {
        let expr = is_nan(get_item("name", root()));
        assert_eq!(expr.to_string(), "vortex.is_nan($.name)");

        let expr2 = is_nan(root());
        assert_eq!(expr2.to_string(), "vortex.is_nan($)");
    }

    #[test]
    fn test_is_nan_is_strict() {
        assert!(
            is_nan(root())
                .as_scalar()
                .is_some_and(|f| f.signature().is_strict())
        );
    }
}

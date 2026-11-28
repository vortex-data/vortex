// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::unstable::row::RowFn;
use vortex_array::scalar_fn::unstable::row::RowVisitor;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

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
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_expr::ArrayExprExt;
    use vortex_expr::get_item;
    use vortex_expr::lit;
    use vortex_expr::root;
    use vortex_session::VortexSession;

    use crate::expr::is_nan;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_array::initialize(&session);
        crate::initialize(&session);
        session
    });

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
    fn evaluate_floats() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array = PrimitiveArray::from_option_iter([
            Some(1.0f32),
            Some(f32::NAN),
            None,
            Some(f32::NEG_INFINITY),
            Some(-f32::NAN),
        ])
        .into_array();

        let result = test_array.apply(&is_nan(root()))?;

        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(false), Some(true), None, Some(false), Some(true)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn evaluate_all_valid_floats() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array = PrimitiveArray::new(
            buffer![1.0f64, f64::NAN, f64::INFINITY, -f64::NAN],
            Validity::NonNullable,
        )
        .into_array();

        let result = test_array.apply(&is_nan(root()))?;

        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false, true, false, true]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn evaluate_all_null_floats() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array = PrimitiveArray::from_option_iter([None::<f32>, None, None]).into_array();

        let result = test_array.apply(&is_nan(root()))?;

        assert_arrays_eq!(result, BoolArray::from_iter([None, None, None]), &mut ctx);
        Ok(())
    }

    #[test]
    fn evaluate_constant() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array = buffer![1.0f32, 2.0, 3.0].into_array();
        let cases = [
            (lit(f32::NAN), BoolArray::from_iter([true, true, true])),
            (lit(1.0f32), BoolArray::from_iter([false, false, false])),
            (
                lit(Scalar::null(DType::Primitive(
                    PType::F32,
                    Nullability::Nullable,
                ))),
                BoolArray::from_iter([None, None, None]),
            ),
        ];

        for (expr_child, expected) in cases {
            let result = test_array.clone().apply(&is_nan(expr_child))?;
            assert_arrays_eq!(result, expected, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn evaluate_rejects_non_float() {
        let test_array = buffer![1i32, 2, 3].into_array();
        assert!(test_array.apply(&is_nan(root())).is_err());
    }

    #[test]
    fn evaluate_sliced() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array = buffer![1.0f32, f32::NAN, 2.0, f32::NAN, 3.0]
            .into_array()
            .slice(1..4)?;

        let result = test_array.apply(&is_nan(root()))?;

        assert_arrays_eq!(result, BoolArray::from_iter([true, false, true]), &mut ctx);
        Ok(())
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

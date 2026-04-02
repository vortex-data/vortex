// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::fmt::Formatter;

pub use kernel::*;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::AnyColumnar;
use crate::ArrayRef;
use crate::CanonicalView;
use crate::ColumnarView;
use crate::ExecutionCtx;
use crate::arrays::Bool;
use crate::arrays::Decimal;
use crate::arrays::Primitive;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// An expression that replaces null values in the input with a fill value.
#[derive(Clone)]
pub struct FillNull;

impl ScalarFnVTable for FillNull {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.fill_null")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            1 => ChildName::from("fill_value"),
            _ => unreachable!("Invalid child index {} for FillNull expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "fill_null(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            arg_dtypes[0].eq_ignore_nullability(&arg_dtypes[1]),
            "fill_null requires input and fill value to have the same base type, got {} and {}",
            arg_dtypes[0],
            arg_dtypes[1]
        );
        // The result dtype takes the nullability of the fill value.
        Ok(arg_dtypes[0]
            .clone()
            .with_nullability(arg_dtypes[1].nullability()))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let fill_value = args.get(1)?;

        let fill_scalar = fill_value
            .as_constant()
            .ok_or_else(|| vortex_err!("fill_null fill_value must be a constant/scalar"))?;

        vortex_ensure!(
            !fill_scalar.is_null(),
            "fill_null requires a non-null fill value"
        );

        let Some(columnar) = input.as_opt::<AnyColumnar>() else {
            return input.execute::<ArrayRef>(ctx)?.fill_null(fill_scalar);
        };

        match columnar {
            ColumnarView::Canonical(canonical) => fill_null_canonical(canonical, &fill_scalar, ctx),
            ColumnarView::Constant(constant) => fill_null_constant(constant, &fill_scalar),
        }
    }

    fn simplify(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        ctx: &dyn crate::scalar_fn::SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        let input_dtype = ctx.return_dtype(expr.child(0))?;

        if !input_dtype.is_nullable() {
            return Ok(Some(expr.child(0).clone()));
        }

        Ok(None)
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        // After fill_null, the result validity depends on the fill value's nullability.
        // If fill_value is non-nullable, the result is always valid.
        Ok(Some(expression.child(1).validity()?))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Fill nulls on a canonical array by directly dispatching to the appropriate kernel.
///
/// Returns the filled array, or bails if no kernel is registered for the canonical type.
fn fill_null_canonical(
    canonical: CanonicalView<'_>,
    fill_value: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let arr = canonical.to_array_ref();
    if let Some(result) = precondition(&arr, fill_value)? {
        // The result of precondition may return another ScalarFn, in which case we should
        // apply it immediately.
        // TODO(aduffy): Remove this once we have better driver check. We're also implicitly
        //  relying on the fact that Cast execution will do an optimize on its result.
        return result.execute::<ArrayRef>(ctx);
    }
    match canonical {
        CanonicalView::Bool(a) => <Bool as FillNullKernel>::fill_null(a, fill_value, ctx)?
            .ok_or_else(|| vortex_err!("FillNullKernel for BoolArray returned None")),
        CanonicalView::Primitive(a) => {
            <Primitive as FillNullKernel>::fill_null(a, fill_value, ctx)?
                .ok_or_else(|| vortex_err!("FillNullKernel for PrimitiveArray returned None"))
        }
        CanonicalView::Decimal(a) => <Decimal as FillNullKernel>::fill_null(a, fill_value, ctx)?
            .ok_or_else(|| vortex_err!("FillNullKernel for DecimalArray returned None")),
        other => vortex_bail!(
            "No FillNullKernel for canonical array {}",
            other.to_array_ref().encoding_id()
        ),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::fill_null;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::root;

    #[test]
    fn dtype() {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        assert_eq!(
            fill_null(root(), lit(0i32)).return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
    }

    #[test]
    fn replace_children() {
        let expr = fill_null(root(), lit(0i32));
        expr.with_children(vec![root(), lit(0i32)])
            .vortex_expect("operation should succeed in test");
    }

    #[test]
    fn evaluate() {
        let test_array =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)])
                .into_array();

        let expr = fill_null(root(), lit(42i32));
        let result = test_array.apply(&expr).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 42, 3, 42, 5]));
    }

    #[test]
    fn evaluate_struct_field() {
        let test_array = StructArray::from_fields(&[(
            "a",
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
        )])
        .unwrap()
        .into_array();

        let expr = fill_null(get_item("a", root()), lit(0i32));
        let result = test_array.apply(&expr).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 0, 3]));
    }

    #[test]
    fn evaluate_non_nullable_input() {
        let test_array = buffer![1i32, 2, 3].into_array();
        let expr = fill_null(root(), lit(0i32));
        let result = test_array.apply(&expr).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 2, 3]));
    }

    #[test]
    fn test_display() {
        let expr = fill_null(get_item("value", root()), lit(0i32));
        assert_eq!(expr.to_string(), "fill_null($.value, 0i32)");
    }
}

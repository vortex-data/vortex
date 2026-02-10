// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::compute::cast;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::EmptyOptions;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::VTable;
use crate::expr::VTableExt;

/// An expression that replaces null values in the input with a fill value.
pub struct FillNull;

impl VTable for FillNull {
    type Options = EmptyOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.fill_null")
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

    fn execute(&self, _options: &Self::Options, args: ExecutionArgs) -> VortexResult<ArrayRef> {
        let len = args.row_count;
        let [input, fill_value]: [ArrayRef; _] = args
            .inputs
            .try_into()
            .map_err(|_| vortex_err!("Wrong arg count"))?;

        let fill_scalar = fill_value
            .as_constant()
            .ok_or_else(|| vortex_err!("fill_null fill_value must be a constant/scalar"))?;

        // If the input has no nulls, fill_null is a no-op (just a cast for nullability).
        if !input.dtype().is_nullable() || input.all_valid()? {
            return cast(input.as_ref(), fill_scalar.dtype());
        }

        // If all values are null, replace the entire array with the fill value.
        if input.all_invalid()? {
            return Ok(ConstantArray::new(fill_scalar, len).into_array());
        }

        // Execute the input child to get it closer to canonical form, then rewrap
        // in a new ScalarFnArray for another optimization round.
        static FILL_NULL_VTABLE: FillNull = FillNull;
        let executed = input.execute::<ArrayRef>(args.ctx)?;
        Ok(ScalarFnArray::try_new(
            crate::expr::ScalarFn::new_static(&FILL_NULL_VTABLE, EmptyOptions),
            vec![executed, fill_value],
            len,
        )?
        .into_array())
    }

    fn simplify(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        ctx: &dyn crate::expr::SimplifyCtx,
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

/// Creates an expression that replaces null values with a fill value.
///
/// ```rust
/// # use vortex_array::expr::{fill_null, root, lit};
/// let expr = fill_null(root(), lit(0i32));
/// ```
pub fn fill_null(child: Expression, fill_value: Expression) -> Expression {
    FillNull.new_expr(EmptyOptions, [child, fill_value])
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_error::VortexExpect;

    use super::fill_null;
    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::root::root;

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

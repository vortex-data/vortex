// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;
use std::fmt::Formatter;

pub use kernel::*;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::mask_validity_canonical;
use crate::builtins::ArrayBuiltins;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::EmptyOptions;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::Literal;
use crate::expr::SimplifyCtx;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::and;
use crate::expr::lit;
use crate::scalar::Scalar;

/// An expression that masks an input based on a boolean mask.
///
/// Where the mask is true, the input value is retained; where the mask is false, the output is
/// null. In other words, this performs an intersection of the input's validity with the mask.
pub struct Mask;

impl VTable for Mask {
    type Options = EmptyOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.mask")
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
            1 => ChildName::from("mask"),
            _ => unreachable!("Invalid child index {} for Mask expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "mask(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            arg_dtypes[1] == DType::Bool(Nullability::NonNullable),
            "The mask argument to 'mask' must be a non-nullable boolean array, got {}",
            arg_dtypes[1]
        );
        Ok(arg_dtypes[0].as_nullable())
    }

    fn execute(&self, _options: &Self::Options, args: ExecutionArgs) -> VortexResult<ArrayRef> {
        let [input, mask_array]: [ArrayRef; _] = args
            .inputs
            .try_into()
            .map_err(|_| vortex_err!("Wrong arg count"))?;

        if let Some(result) = execute_constant(&input, &mask_array)? {
            return Ok(result);
        }

        execute_canonical(input, mask_array, args.ctx)
    }

    fn simplify(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        let Some(mask_lit) = expr.child(1).as_opt::<Literal>() else {
            return Ok(None);
        };

        let mask_lit = mask_lit
            .as_bool()
            .value()
            .vortex_expect("Mask must be non-nullable");

        if mask_lit {
            // Mask is all true, so the output is just the input.
            Ok(Some(expr.child(0).clone()))
        } else {
            // Mask is all false, so the output is all nulls.
            let input_dtype = ctx.return_dtype(expr.child(0))?;
            Ok(Some(lit(Scalar::null(input_dtype.as_nullable()))))
        }
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(and(
            expression.child(0).validity()?,
            expression.child(1).clone(),
        )))
    }
}

/// Try to handle masking when at least one of the input or mask is a constant array.
///
/// Returns `Ok(Some(result))` if the constant case was handled, `Ok(None)` if not.
fn execute_constant(input: &ArrayRef, mask_array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
    let len = input.len();

    if let Some(constant_mask) = mask_array.as_opt::<ConstantVTable>() {
        let mask_value = constant_mask.scalar().as_bool().value().unwrap_or(false);
        return if mask_value {
            input.cast(input.dtype().as_nullable()).map(Some)
        } else {
            Ok(Some(
                ConstantArray::new(Scalar::null(input.dtype().as_nullable()), len).into_array(),
            ))
        };
    }

    if let Some(constant_input) = input.as_opt::<ConstantVTable>()
        && constant_input.scalar().is_null()
    {
        return Ok(Some(
            ConstantArray::new(Scalar::null(input.dtype().as_nullable()), len).into_array(),
        ));
    }

    Ok(None)
}

/// Execute the mask by materializing both inputs to their canonical forms.
fn execute_canonical(
    input: ArrayRef,
    mask_array: ArrayRef,
    ctx: &mut crate::executor::ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mask_bool = mask_array.execute::<BoolArray>(ctx)?;
    let validity_mask = vortex_mask::Mask::from(mask_bool.to_bit_buffer());

    let canonical = input.execute::<Canonical>(ctx)?;
    Ok(mask_validity_canonical(canonical, &validity_mask, ctx)?.into_array())
}

/// Creates a mask expression that applies the given boolean mask to the input array.
pub fn mask(array: Expression, mask: Expression) -> Expression {
    Mask.new_expr(EmptyOptions, [array, mask])
}

#[cfg(test)]
mod test {
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::Nullable;
    use vortex_dtype::PType;
    use vortex_error::VortexExpect;

    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::mask::mask;
    use crate::scalar::Scalar;

    #[test]
    fn test_simplify() {
        let input_expr = lit(42u32);
        let true_mask_expr = lit(true);
        let false_mask_expr = lit(false);

        let mask_true_expr = mask(input_expr.clone(), true_mask_expr);
        let simplified_true = mask_true_expr
            .optimize(&DType::Null)
            .vortex_expect("Simplification");
        assert_eq!(&simplified_true, &input_expr);

        let mask_false_expr = mask(input_expr, false_mask_expr);
        let simplified_false = mask_false_expr
            .optimize(&DType::Null)
            .vortex_expect("Simplification");
        let expected_null_expr = lit(Scalar::null(DType::Primitive(PType::U32, Nullable)));
        assert_eq!(&simplified_false, &expected_null_expr);
    }
}

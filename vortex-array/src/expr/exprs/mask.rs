// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::ops::Not;

use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_scalar::Scalar;
use vortex_vector::BoolDatum;
use vortex_vector::Datum;
use vortex_vector::ScalarOps;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;

use crate::Array;
use crate::ArrayRef;
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

    fn deserialize(&self, _metadata: &[u8]) -> VortexResult<Self::Options> {
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

    fn evaluate(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        scope: &ArrayRef,
    ) -> VortexResult<ArrayRef> {
        let child = expr.child(0).evaluate(scope)?;

        // Invert the validity mask - we want to set values to null where validity is false.
        let inverted_mask = child.validity_mask().not();

        crate::compute::mask(&child, &inverted_mask)
    }

    fn execute(&self, _options: &Self::Options, args: ExecutionArgs) -> VortexResult<Datum> {
        let [input, mask]: [Datum; _] = args
            .datums
            .try_into()
            .map_err(|_| vortex_err!("Wrong arg count"))?;
        let mask = mask.into_bool();

        match (input, mask) {
            (Datum::Scalar(input), BoolDatum::Scalar(mask)) => {
                let mut result = input;
                result.mask_validity(mask.value().vortex_expect("mask is non-nullable"));
                Ok(Datum::Scalar(result))
            }
            (Datum::Scalar(input), BoolDatum::Vector(mask)) => {
                let mut result = input.repeat(args.row_count).freeze();
                result.mask_validity(&vortex_mask::Mask::from(mask.into_bits()));
                Ok(Datum::Vector(result))
            }
            (Datum::Vector(input_array), BoolDatum::Scalar(mask)) => {
                let mut result = input_array;
                result.mask_validity(&vortex_mask::Mask::new(
                    args.row_count,
                    mask.value().vortex_expect("mask is non-nullable"),
                ));
                Ok(Datum::Vector(result))
            }
            (Datum::Vector(input_array), BoolDatum::Vector(mask)) => {
                let mut result = input_array;
                result.mask_validity(&vortex_mask::Mask::from(mask.into_bits()));
                Ok(Datum::Vector(result))
            }
        }
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
    use vortex_scalar::Scalar;

    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::mask::mask;

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

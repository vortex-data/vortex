// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::fmt::Formatter;

pub use kernel::*;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::builtins::ArrayBuiltins;
use crate::compute::zip_impl;
use crate::compute::zip_return_dtype;
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

/// An expression that conditionally selects between two arrays based on a boolean mask.
///
/// For each position `i`, `result[i] = if mask[i] then if_true[i] else if_false[i]`.
///
/// Null values in the mask are treated as false (selecting `if_false`). This follows
/// SQL semantics (DuckDB, Trino) where a null condition falls through to the ELSE branch,
/// rather than Arrow's `if_else` which propagates null conditions to the output.
pub struct Zip;

impl VTable for Zip {
    type Options = EmptyOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.zip")
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
        Arity::Exact(3)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("if_true"),
            1 => ChildName::from("if_false"),
            2 => ChildName::from("mask"),
            _ => unreachable!("Invalid child index {} for Zip expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "zip(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(2).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            arg_dtypes[0].eq_ignore_nullability(&arg_dtypes[1]),
            "zip requires if_true and if_false to have the same base type, got {} and {}",
            arg_dtypes[0],
            arg_dtypes[1]
        );
        vortex_ensure!(
            matches!(arg_dtypes[2], DType::Bool(_)),
            "zip requires mask to be a boolean type, got {}",
            arg_dtypes[2]
        );
        Ok(arg_dtypes[0]
            .clone()
            .union_nullability(arg_dtypes[1].nullability()))
    }

    fn execute(&self, _options: &Self::Options, args: ExecutionArgs) -> VortexResult<ArrayRef> {
        let [if_true, if_false, mask_array]: [ArrayRef; _] = args
            .inputs
            .try_into()
            .map_err(|_| vortex_err!("Wrong arg count"))?;

        let mask = mask_array.try_to_mask_fill_null_false()?;

        if mask.all_true() {
            return if_true
                .cast(zip_return_dtype(&if_true, &if_false))?
                .execute(args.ctx);
        }

        if mask.all_false() {
            return if_false
                .cast(zip_return_dtype(&if_true, &if_false))?
                .execute(args.ctx);
        }

        if !if_true.is_canonical() || !if_false.is_canonical() {
            let if_true = if_true.execute::<ArrayRef>(args.ctx)?;
            let if_false = if_false.execute::<ArrayRef>(args.ctx)?;
            return crate::compute::zip(&if_true, &if_false, &mask);
        }

        zip_impl(&if_true, &if_false, &mask)
    }

    fn simplify(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        _ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        let Some(mask_lit) = expr.child(2).as_opt::<Literal>() else {
            return Ok(None);
        };

        if let Some(mask_val) = mask_lit.as_bool().value() {
            if mask_val {
                return Ok(Some(expr.child(0).clone()));
            } else {
                return Ok(Some(expr.child(1).clone()));
            }
        }

        Ok(None)
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Creates a zip expression that conditionally selects between two arrays.
///
/// ```rust
/// # use vortex_array::expr::{zip_expr, root, lit};
/// let expr = zip_expr(root(), lit(0i32), lit(true));
/// ```
pub fn zip_expr(if_true: Expression, if_false: Expression, mask: Expression) -> Expression {
    Zip.new_expr(EmptyOptions, [if_true, if_false, mask])
}

#[cfg(test)]
mod tests {
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;

    use super::zip_expr;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::root::root;

    #[test]
    fn dtype() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let expr = zip_expr(root(), lit(0i32), lit(true));
        let result_dtype = expr.return_dtype(&dtype).unwrap();
        assert_eq!(
            result_dtype,
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_display() {
        let expr = zip_expr(root(), lit(0i32), lit(true));
        assert_eq!(expr.to_string(), "zip($, 0i32, true)");
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;

use arbitrary::Result as AResult;
use arbitrary::Unstructured;
use vortex_dtype::DType;
use vortex_dtype::FieldName;
use vortex_scalar::arbitrary::random_scalar;

use crate::expr::Binary;
use crate::expr::Expression;
use crate::expr::Operator;
use crate::expr::VTableExt;
use crate::expr::and_collect;
use crate::expr::col;
use crate::expr::lit;
use crate::expr::pack;
use crate::expr::root;

/// Context for arbitrary expression generation.
/// Provides recursive generation capabilities for building well-typed expression trees.
pub trait ArbExprCtx {
    /// Bottom-up: wrap `child` (containing root) with a random expression.
    /// Returns the wrapped expression and its output dtype.
    fn grow(
        &self,
        u: &mut Unstructured,
        scope: &DType,
        child: Expression,
        child_type: &DType,
        depth: u8,
    ) -> AResult<(Expression, DType)>;

    /// Top-down: generate an expression producing `target` dtype.
    /// No root requirement - used for "other" children in multi-child expressions.
    fn generate(
        &self,
        u: &mut Unstructured,
        scope: &DType,
        target: &DType,
        depth: u8,
    ) -> AResult<Option<Expression>>;
}

/// Trait for expression vtables to support arbitrary generation.
/// Implement this separately from the main VTable trait.
pub trait ArbExpr: 'static + Send + Sync {
    /// Bottom-up: try to wrap `child` as one of this expression's children.
    /// Generate any additional children via `ctx.generate()`.
    /// Returns `Ok(None)` if this expression cannot wrap the given child type.
    fn arb_wrap(
        &self,
        u: &mut Unstructured,
        scope: &DType,
        child: Expression,
        child_type: &DType,
        ctx: &dyn ArbExprCtx,
    ) -> AResult<Option<(Expression, DType)>>;

    /// Top-down: try to generate an expression producing `target` dtype.
    /// Returns `Ok(None)` if this expression cannot produce the target type.
    fn arb_gen(
        &self,
        u: &mut Unstructured,
        scope: &DType,
        target: &DType,
        depth: u8,
        ctx: &dyn ArbExprCtx,
    ) -> AResult<Option<Expression>>;
}

/// Default context implementation that holds registered ArbExpr implementations.
pub struct ArbExprCtxImpl {
    arb_exprs: Vec<&'static dyn ArbExpr>,
}

impl Default for ArbExprCtxImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl ArbExprCtxImpl {
    /// Create a new context with all built-in expression generators.
    pub fn new() -> Self {
        use crate::expr::exprs::binary::Binary;
        use crate::expr::exprs::get_item::GetItem;
        use crate::expr::exprs::is_null::IsNull;
        use crate::expr::exprs::literal::Literal;
        use crate::expr::exprs::not::Not;
        use crate::expr::exprs::root::Root;

        Self {
            arb_exprs: vec![&Root, &Literal, &Binary, &GetItem, &Not, &IsNull],
        }
    }
}

impl ArbExprCtx for ArbExprCtxImpl {
    fn grow(
        &self,
        u: &mut Unstructured,
        scope: &DType,
        child: Expression,
        child_type: &DType,
        depth: u8,
    ) -> AResult<(Expression, DType)> {
        if depth == 0 {
            return Ok((child, child_type.clone()));
        }

        // Try wrappers starting from a random index
        let len = self.arb_exprs.len();
        let start = u.int_in_range(0..=len - 1)?;

        for offset in 0..len {
            let i = (start + offset) % len;
            if let Some((wrapped, dtype)) =
                self.arb_exprs[i].arb_wrap(u, scope, child.clone(), child_type, self)?
            {
                return Ok((wrapped, dtype));
            }
        }

        // No wrapper found, return child as-is
        Ok((child, child_type.clone()))
    }

    fn generate(
        &self,
        u: &mut Unstructured,
        scope: &DType,
        target: &DType,
        depth: u8,
    ) -> AResult<Option<Expression>> {
        // Try generators starting from a random index
        let len = self.arb_exprs.len();
        let start = u.int_in_range(0..=len - 1)?;

        for offset in 0..len {
            let i = (start + offset) % len;
            if let Some(expr) = self.arb_exprs[i].arb_gen(u, scope, target, depth, self)? {
                return Ok(Some(expr));
            }
        }

        Ok(None)
    }
}

/// Generate an arbitrary expression of any type, guaranteed to contain `root()`.
///
/// Starts from `root()` and builds up by wrapping with random expressions.
pub fn arb_expr(u: &mut Unstructured, scope: &DType, depth: u8) -> AResult<(Expression, DType)> {
    let ctx = ArbExprCtxImpl::new();
    let (mut expr, mut dtype) = (root(), scope.clone());

    for _ in 0..depth {
        let (new_expr, new_dtype) = ctx.grow(u, scope, expr.clone(), &dtype, 1)?;
        if new_expr == expr {
            break; // No progress
        }
        expr = new_expr;
        dtype = new_dtype;
    }

    Ok((expr, dtype))
}

/// Generate an arbitrary filter expression (returns Bool), guaranteed to contain `root()`.
///
/// Starts from `root()` and builds up until we produce a Bool type.
pub fn arb_filter_expr(
    u: &mut Unstructured,
    scope: &DType,
    depth: u8,
) -> AResult<Option<Expression>> {
    let ctx = ArbExprCtxImpl::new();
    let (mut expr, mut dtype) = (root(), scope.clone());

    // Grow until we produce Bool
    for _ in 0..depth {
        if matches!(dtype, DType::Bool(_)) {
            // Randomly stop or continue
            if u.ratio(1, 3)? {
                break;
            }
        }
        let (new_expr, new_dtype) = ctx.grow(u, scope, expr.clone(), &dtype, 1)?;
        if new_expr == expr {
            break; // No progress
        }
        expr = new_expr;
        dtype = new_dtype;
    }

    if matches!(dtype, DType::Bool(_)) {
        Ok(Some(expr))
    } else {
        Ok(None)
    }
}

pub fn projection_expr(u: &mut Unstructured<'_>, dtype: &DType) -> AResult<Option<Expression>> {
    let Some(struct_dtype) = dtype.as_struct_fields_opt() else {
        return Ok(None);
    };

    let column_count = u.int_in_range::<usize>(0..=max(struct_dtype.nfields(), 10))?;

    let cols = (0..column_count)
        .map(|_| {
            let get_item = u.choose_iter(struct_dtype.names().iter())?;
            Ok((get_item.clone(), col(get_item.clone())))
        })
        .collect::<AResult<Vec<_>>>()?;

    Ok(Some(pack(cols, u.arbitrary()?)))
}

pub fn filter_expr(u: &mut Unstructured<'_>, dtype: &DType) -> AResult<Option<Expression>> {
    let Some(struct_dtype) = dtype.as_struct_fields_opt() else {
        return Ok(None);
    };

    let filter_count = u.int_in_range::<usize>(0..=max(struct_dtype.nfields(), 10))?;

    let filters = (0..filter_count)
        .map(|_| {
            let (col, dtype) =
                u.choose_iter(struct_dtype.names().iter().zip(struct_dtype.fields()))?;
            random_comparison(u, col, &dtype)
        })
        .collect::<AResult<Vec<_>>>()?;

    Ok(and_collect(filters))
}

fn random_comparison(
    u: &mut Unstructured<'_>,
    name: &FieldName,
    dtype: &DType,
) -> AResult<Expression> {
    let scalar = random_scalar(u, dtype)?;
    Ok(Binary.new_expr(
        arbitrary_comparison_operator(u)?,
        [col(name.clone()), lit(scalar)],
    ))
}

fn arbitrary_comparison_operator(u: &mut Unstructured<'_>) -> AResult<Operator> {
    Ok(match u.int_in_range(0..=5)? {
        0 => Operator::Eq,
        1 => Operator::NotEq,
        2 => Operator::Gt,
        3 => Operator::Gte,
        4 => Operator::Lt,
        5 => Operator::Lte,
        _ => unreachable!("range 0..=5"),
    })
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::RefCell;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::transform::match_between::find_between;
use crate::scalar_fn::ExpressionReduceNode;
use crate::scalar_fn::SimplifyCtx;

impl Expression {
    /// Optimize the root expression node only, iterating to convergence.
    ///
    /// This applies optimization rules repeatedly until no more changes occur:
    /// 1. `simplify_untyped` - type-independent simplifications
    /// 2. `simplify` - type-aware simplifications
    /// 3. `reduce` - abstract reduction rules via `ReduceNode`
    pub fn optimize(&self, scope: &DType) -> VortexResult<Expression> {
        let cache = SimplifyCache::new(scope);
        Ok(self.try_optimize(&cache)?.unwrap_or_else(|| self.clone()))
    }

    /// Apply this node's own untyped simplification rule, if it has one.
    ///
    /// Non-scalar nodes carry no rules, so they never simplify.
    fn simplify_untyped_node(&self) -> VortexResult<Option<Expression>> {
        match self {
            Expression::Scalar { scalar_fn, .. } => scalar_fn.simplify_untyped(self),
            Expression::Root => Ok(None),
        }
    }

    /// Apply this node's own type-aware simplification rule, if it has one.
    fn simplify_node(&self, ctx: &dyn SimplifyCtx) -> VortexResult<Option<Expression>> {
        match self {
            Expression::Scalar { scalar_fn, .. } => scalar_fn.simplify(self, ctx),
            Expression::Root => Ok(None),
        }
    }

    /// Apply this node's own abstract reduction rule, if it has one.
    fn reduce_node<'a>(
        &self,
        node: &ExpressionReduceNode<'a>,
    ) -> VortexResult<Option<ExpressionReduceNode<'a>>> {
        match self {
            Expression::Scalar { scalar_fn, .. } => scalar_fn.reduce_expression(node),
            Expression::Root => Ok(None),
        }
    }

    /// Try to optimize the root expression node only, returning None if no optimizations applied.
    fn try_optimize(&self, cache: &SimplifyCache<'_>) -> VortexResult<Option<Expression>> {
        // Copy-on-write: `current` stays None until a rule fires, so unchanged nodes (the common
        // case) are never cloned.
        let mut current: Option<Expression> = None;
        let mut loop_counter = 0;

        loop {
            if loop_counter > 100 {
                vortex_error::vortex_bail!(
                    "Exceeded maximum optimization iterations (possible infinite loop)"
                );
            }
            loop_counter += 1;

            let expr = current.as_ref().unwrap_or(self);
            let mut changed = false;

            // Try simplify_untyped
            if let Some(simplified) = expr.simplify_untyped_node()? {
                current = Some(simplified);
                changed = true;
            }

            // Try simplify (typed)
            let expr = current.as_ref().unwrap_or(self);
            if let Some(simplified) = expr.simplify_node(cache)? {
                current = Some(simplified);
                changed = true;
            }

            // Try reduce via ReduceNode. The node borrows the expression and scope, so
            // constructing it is free; the block scopes the borrows so `current` can be updated.
            let reduced = {
                let expr = current.as_ref().unwrap_or(self);
                let reduce_node = ExpressionReduceNode::new(expr, cache.scope);
                expr.reduce_node(&reduce_node)?
                    .map(ExpressionReduceNode::into_expression)
            };
            if let Some(reduced_expr) = reduced {
                current = Some(reduced_expr);
                changed = true;
            }

            if !changed {
                break;
            }
        }

        Ok(current)
    }

    /// Optimize the entire expression tree recursively.
    ///
    /// Optimizes children first (bottom-up), then optimizes the root.
    pub fn optimize_recursive(&self, scope: &DType) -> VortexResult<Expression> {
        Ok(self
            .clone()
            .try_optimize_recursive(scope)?
            .unwrap_or_else(|| self.clone()))
    }

    /// Try to optimize the entire expression tree recursively.
    pub fn try_optimize_recursive(&self, scope: &DType) -> VortexResult<Option<Expression>> {
        let cache = SimplifyCache::new(scope);
        let result = self.try_optimize_recursive_inner(&cache)?;

        // Apply the between optimization once at the top level only.
        // TODO(ngates): remove the "between" optimization, or rewrite it to not always convert
        //  to CNF?
        Ok(Some(find_between(result.unwrap_or_else(|| self.clone()))))
    }

    fn try_optimize_recursive_inner(
        &self,
        cache: &SimplifyCache<'_>,
    ) -> VortexResult<Option<Expression>> {
        // First optimize the root
        let mut current = self.try_optimize(cache)?;

        // Then recursively optimize children. The new children vector is only allocated once a
        // child actually changes, so fully-optimized subtrees cost no allocations.
        let expr = current.as_ref().unwrap_or(self);
        let children = expr.children();
        let mut new_children: Option<Vec<Expression>> = None;
        for (idx, child) in children.iter().enumerate() {
            if let Some(optimized) = child.try_optimize_recursive_inner(cache)? {
                new_children
                    .get_or_insert_with(|| children[..idx].to_vec())
                    .push(optimized);
            } else if let Some(new_children) = new_children.as_mut() {
                new_children.push(child.clone());
            }
        }

        if let Some(new_children) = new_children {
            let updated = expr.clone().with_children(new_children)?;

            // After updating children, try to optimize root again
            current = Some(updated.try_optimize(cache)?.unwrap_or(updated));
        }

        Ok(current)
    }
}

struct SimplifyCache<'a> {
    scope: &'a DType,
    dtype_cache: RefCell<HashMap<Expression, DType>>,
}

impl<'a> SimplifyCache<'a> {
    fn new(scope: &'a DType) -> Self {
        Self {
            scope,
            dtype_cache: RefCell::new(HashMap::new()),
        }
    }
}

impl SimplifyCtx for SimplifyCache<'_> {
    fn return_dtype(&self, expr: &Expression) -> VortexResult<DType> {
        // If the expression is "root", return the scope dtype
        if expr.is_root() {
            return Ok(self.scope.clone());
        }

        if let Some(dtype) = self.dtype_cache.borrow().get(expr) {
            return Ok(dtype.clone());
        }

        // Otherwise, compute dtype from children
        let input_dtypes: Vec<_> = expr
            .children()
            .iter()
            .map(|c| self.return_dtype(c))
            .try_collect()?;
        let dtype = expr
            .as_scalar()
            .ok_or_else(|| vortex_err!("cannot type a non-scalar expression: {expr}"))?
            .return_dtype(&input_dtypes)?;
        self.dtype_cache
            .borrow_mut()
            .insert(expr.clone(), dtype.clone());

        Ok(dtype)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use crate::dtype::DType;
    use crate::dtype::FieldName;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::cast;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::lt_eq;
    use crate::expr::merge;
    use crate::expr::or;
    use crate::expr::pack;
    use crate::expr::root;
    use crate::expr::select;
    use crate::expr::transform::replace;
    use crate::expr::transform::replace_root_fields;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::literal::Literal;

    #[test]
    fn optimize_or_chain_correctness() -> VortexResult<()> {
        let expr = or(
            eq(get_item("x", root()), lit(1i32)),
            eq(get_item("x", root()), lit(2i32)),
        );
        let scope = DType::Struct(
            StructFields::new(
                ["x"].into(),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let optimized = expr.optimize_recursive(&scope)?;

        let s = optimized.to_string();
        assert!(s.contains("$.x"), "expected $.x in {s}");
        assert!(s.contains("1i32") || s.contains('1'), "expected 1 in {s}");
        assert!(s.contains("2i32") || s.contains('2'), "expected 2 in {s}");
        Ok(())
    }

    #[test]
    fn optimize_folds_cast_of_literal_in_comparison() -> VortexResult<()> {
        let expr = lt_eq(
            get_item("x", root()),
            cast(
                lit(3i32),
                DType::Primitive(PType::F64, Nullability::NonNullable),
            ),
        );
        let scope = DType::Struct(
            StructFields::new(
                ["x"].into(),
                vec![DType::Primitive(PType::F64, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let optimized = expr.optimize_recursive(&scope)?;

        // Prune rules pattern-match a bare Literal on the comparison RHS; a cast wrapper
        // silently disables pruning.
        let rhs = optimized
            .child(1)
            .as_opt::<Literal>()
            .ok_or_else(|| vortex_err!("expected a bare literal RHS, got {optimized}"))?;
        assert_eq!(rhs, &Scalar::primitive(3.0f64, Nullability::NonNullable));
        Ok(())
    }

    /// `Select` over a `pack` rewrites to a `pack` of the selected child fields. It must read
    /// those fields out of the child `pack` itself, because the optimizer visits children before
    /// their parent and so never revisits the children a rule introduces.
    #[test]
    fn optimize_select_of_pack_reads_pack_fields() -> VortexResult<()> {
        let field_names = (0..4)
            .map(|idx| FieldName::from(format!("field_{idx}")))
            .collect::<Vec<_>>();
        let fields = StructFields::new(
            field_names.clone().into(),
            vec![DType::Primitive(PType::U64, Nullability::NonNullable); field_names.len()],
        );
        let scope = DType::Struct(fields.clone(), Nullability::NonNullable);

        let expr = select(
            field_names
                .iter()
                .cloned()
                .chain(["input_row_idx".into(), "input_file_idx".into()])
                .collect::<Vec<_>>(),
            merge([
                root(),
                pack(
                    [
                        ("input_row_idx", lit(0_u64)),
                        ("input_file_idx", lit(1_u64)),
                    ],
                    Nullability::NonNullable,
                ),
            ]),
        );
        let expr = replace(expr, &root(), replace_root_fields(root(), &fields));

        let expected = pack(
            field_names
                .into_iter()
                .map(|name| (name.clone(), get_item(name, root())))
                .chain([
                    ("input_row_idx".into(), lit(0_u64)),
                    ("input_file_idx".into(), lit(1_u64)),
                ])
                .collect::<Vec<_>>(),
            Nullability::NonNullable,
        );

        let optimized = expr.optimize_recursive(&scope)?;
        assert_eq!(optimized, expected, "got {optimized}");
        assert_eq!(optimized.optimize_recursive(&scope)?, expected);

        Ok(())
    }

    /// `Merge` reduces to a `pack` that reads each field out of its children, so it must also
    /// read straight out of a `pack` child.
    #[test]
    fn optimize_merge_of_packs_reads_pack_fields() -> VortexResult<()> {
        let scope = DType::Struct(StructFields::empty(), Nullability::NonNullable);
        let expr = merge([
            pack([("a", lit(0_u64))], Nullability::NonNullable),
            pack([("b", lit(1_u64))], Nullability::NonNullable),
        ]);

        let expected = pack(
            [("a", lit(0_u64)), ("b", lit(1_u64))],
            Nullability::NonNullable,
        );

        let optimized = expr.optimize_recursive(&scope)?;
        assert_eq!(optimized, expected, "got {optimized}");
        assert_eq!(optimized.optimize_recursive(&scope)?, expected);

        Ok(())
    }
}

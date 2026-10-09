// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::expr::BoundExpression;
use crate::expr::transform::match_between::find_between;
use crate::scalar_fn::ExpressionReduceNode;

impl BoundExpression {
    /// Optimize the root expression node only, iterating to convergence.
    ///
    /// This applies optimization rules repeatedly until no more changes occur:
    /// 1. `simplify` - scalar-function simplifications over the bound node
    /// 2. `reduce` - abstract reduction rules via `ReduceNode`
    pub fn optimize(&self) -> VortexResult<BoundExpression> {
        Ok(self.try_optimize()?.unwrap_or_else(|| self.clone()))
    }

    /// Apply this node's own simplification rule, if it has one.
    ///
    /// Non-scalar nodes carry no rules, so they never simplify.
    fn simplify_node(&self) -> VortexResult<Option<BoundExpression>> {
        match self {
            BoundExpression::Scalar { scalar_fn, .. } => scalar_fn.simplify(self),
            BoundExpression::Root { .. } => Ok(None),
        }
    }

    /// Apply this node's own abstract reduction rule, if it has one.
    fn reduce_node<'a>(
        &self,
        node: &ExpressionReduceNode<'a>,
    ) -> VortexResult<Option<ExpressionReduceNode<'a>>> {
        match self {
            BoundExpression::Scalar { scalar_fn, .. } => scalar_fn.reduce_expression(node),
            BoundExpression::Root { .. } => Ok(None),
        }
    }

    /// Try to optimize the root expression node only, returning None if no optimizations applied.
    fn try_optimize(&self) -> VortexResult<Option<BoundExpression>> {
        // Copy-on-write: `current` stays None until a rule fires, so unchanged nodes (the common
        // case) are never cloned.
        let mut current: Option<BoundExpression> = None;
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

            if let Some(simplified) = expr.simplify_node()? {
                current = Some(simplified);
                changed = true;
            }

            // Try reduce via ReduceNode. The node borrows the expression, so constructing it is
            // free; the block scopes the borrow so `current` can be updated.
            let reduced = {
                let expr = current.as_ref().unwrap_or(self);
                expr.reduce_node(&ExpressionReduceNode::new(expr))?
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
    pub fn optimize_recursive(&self) -> VortexResult<BoundExpression> {
        Ok(self
            .try_optimize_recursive()?
            .unwrap_or_else(|| self.clone()))
    }

    /// Try to optimize the entire expression tree recursively.
    pub fn try_optimize_recursive(&self) -> VortexResult<Option<BoundExpression>> {
        let result = self.try_optimize_recursive_inner()?;

        // Apply the between optimization once at the top level only.
        // TODO(ngates): remove the "between" optimization, or rewrite it to not always convert
        //  to CNF?
        Ok(Some(find_between(result.unwrap_or_else(|| self.clone()))))
    }

    fn try_optimize_recursive_inner(&self) -> VortexResult<Option<BoundExpression>> {
        // First optimize the root
        let mut current = self.try_optimize()?;

        // Then recursively optimize children. The new children vector is only allocated once a
        // child actually changes, so fully-optimized subtrees cost no allocations.
        let expr = current.as_ref().unwrap_or(self);
        let children = expr.children();
        let mut new_children: Option<Vec<BoundExpression>> = None;
        for (idx, child) in children.iter().enumerate() {
            if let Some(optimized) = child.try_optimize_recursive_inner()? {
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
            current = Some(updated.try_optimize()?.unwrap_or(updated));
        }

        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use crate::dtype::DType;
    use crate::dtype::FieldName;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::Expression;
    use crate::expr::and;
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
    use crate::expr::zip_expr;
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
        let optimized = expr.bind(&scope)?.optimize_recursive()?;

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
        let optimized = expr.bind(&scope)?.optimize_recursive()?;

        // Prune rules pattern-match a bare Literal on the comparison RHS; a cast wrapper
        // silently disables pruning.
        let rhs = optimized
            .child(1)
            .as_opt::<Literal>()
            .ok_or_else(|| vortex_err!("expected a bare literal RHS, got {optimized}"))?;
        assert_eq!(rhs, &Scalar::primitive(3.0f64, Nullability::NonNullable));
        Ok(())
    }

    #[rstest]
    #[case::and_annihilator(and(root(), lit(false)), DType::Bool(Nullability::Nullable))]
    #[case::or_annihilator(or(root(), lit(true)), DType::Bool(Nullability::Nullable))]
    #[case::and_nullable_identity(
        and(root(), lit(Scalar::from(Some(true)))),
        DType::Bool(Nullability::NonNullable)
    )]
    #[case::or_nullable_identity(
        or(root(), lit(Scalar::from(Some(false)))),
        DType::Bool(Nullability::NonNullable)
    )]
    #[case::zip_true(
        zip_expr(lit(true), lit(1i32), root()),
        DType::Primitive(PType::I32, Nullability::Nullable)
    )]
    #[case::zip_false(
        zip_expr(lit(false), root(), lit(1i32)),
        DType::Primitive(PType::I32, Nullability::Nullable)
    )]
    fn literal_simplification_preserves_result_dtype(
        #[case] expr: Expression,
        #[case] scope: DType,
    ) -> VortexResult<()> {
        let original = expr.bind(&scope)?;
        let optimized = expr.bind(&scope)?.optimize_recursive()?;
        assert_eq!(optimized.dtype(), original.dtype());

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
        let scope = DType::Struct(fields, Nullability::NonNullable);

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
        let root_pack = pack(
            field_names
                .iter()
                .map(|name| (name.clone(), get_item(name.clone(), root())))
                .collect::<Vec<_>>(),
            Nullability::NonNullable,
        );
        let expr = replace(expr, &root(), root_pack);

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

        let expected = expected.bind(&scope)?;

        let optimized = expr.bind(&scope)?.optimize_recursive()?;
        assert_eq!(optimized, expected, "got {optimized}");
        assert_eq!(optimized.optimize_recursive()?, expected);

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

        let expected = expected.bind(&scope)?;

        let optimized = expr.bind(&scope)?.optimize_recursive()?;
        assert_eq!(optimized, expected, "got {optimized}");
        assert_eq!(optimized.optimize_recursive()?, expected);

        Ok(())
    }
}

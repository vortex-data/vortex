// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::expr::BoundExpression;
use crate::expr::transform::match_between::find_between_bound;

impl BoundExpression {
    /// Optimize the root node with simplification and abstract reduction.
    ///
    /// Expressions must be bound before they can be optimized, even for rules that do not inspect
    /// dtypes. This keeps the optimizer's input and output in the same typed representation.
    pub fn optimize(&self) -> VortexResult<BoundExpression> {
        Ok(self.try_optimize()?.unwrap_or_else(|| self.clone()))
    }

    fn simplify_node(&self) -> VortexResult<Option<BoundExpression>> {
        match self {
            BoundExpression::Scalar { scalar_fn, .. } => scalar_fn.simplify(self),
            BoundExpression::Root { .. } => Ok(None),
        }
    }

    fn reduce_node(&self) -> VortexResult<Option<BoundExpression>> {
        match self {
            BoundExpression::Scalar { scalar_fn, .. } => scalar_fn.reduce_bound_expression(self),
            BoundExpression::Root { .. } => Ok(None),
        }
    }

    fn try_optimize(&self) -> VortexResult<Option<BoundExpression>> {
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

            let expr = current.as_ref().unwrap_or(self);
            if let Some(reduced) = expr.reduce_node()? {
                current = Some(reduced);
                changed = true;
            }

            if !changed {
                break;
            }
        }

        Ok(current)
    }

    /// Optimize the entire bound expression tree recursively.
    pub fn optimize_recursive(&self) -> VortexResult<BoundExpression> {
        Ok(self
            .clone()
            .try_optimize_recursive()?
            .unwrap_or_else(|| self.clone()))
    }

    pub fn try_optimize_recursive(&self) -> VortexResult<Option<BoundExpression>> {
        let result = self.try_optimize_recursive_inner()?;

        // Apply the between optimization once at the top level only.
        // TODO(ngates): remove the "between" optimization, or rewrite it to not always convert
        // to CNF?
        Ok(Some(find_between_bound(
            result.unwrap_or_else(|| self.clone()),
        )))
    }

    fn try_optimize_recursive_inner(&self) -> VortexResult<Option<BoundExpression>> {
        let mut current = self.try_optimize()?;

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
            current = Some(updated.try_optimize()?.unwrap_or(updated));
        }

        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::cast;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::lt_eq;
    use crate::expr::or;
    use crate::expr::root;
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
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rule-driven optimization for [`BoundExpression`] trees.
//!
//! This optimizer is independent from the legacy [`Expression`](super::Expression) optimizer.
//! Rules operate only on already-bound expressions, which makes every node's dtype available in
//! constant time and lets the driver verify that rewrites preserve the tree's type proof.
//!
//! # How optimization works
//!
//! [`OptimizerRuleRegistry`] holds the reusable rewrite rules. A
//! [`BoundExpressionOptimizer`] takes ownership of a configured registry and applies its rules
//! with a configurable rewrite limit. Each call to [`BoundExpressionOptimizer::try_optimize`]
//! creates an `OptimizationRun` containing a reference to the registry and the mutable state for
//! that traversal.
//!
//! A run recursively walks the tree with copy-on-write rebuilding:
//!
//! 1. Rules registered for a node's expression ID run in registration order before its children.
//!    The first matching rule wins, and root rewrites repeat to convergence.
//! 2. Children are optimized in evaluation order. The node is rebuilt only if a child changed.
//! 3. A rebuilt node is rewritten again. Any replacement is walked as a new subtree so rules may
//!    safely introduce expressions that need further optimization.
//!
//! After the rule-driven traversal reaches a fixpoint, compatible bounds in the top-level
//! conjunction are combined into `between` expressions.
//!
//! Every replacement must preserve the node's dtype and differ from the expression it replaces.
//! A per-run rewrite limit terminates rule cycles, and a depth limit prevents stack overflow.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::expr::BoundExpression;

mod find_between;
mod rules;

use find_between::find_between;
pub use rules::OptimizerRule;
pub use rules::OptimizerRuleRef;
pub use rules::OptimizerRuleRegistry;

const DEFAULT_MAX_REWRITES: usize = 10_000;
const MAX_DEPTH: usize = 256;

/// A deterministic optimizer for [`BoundExpression`] trees.
///
/// The optimizer uses the rules in its [`OptimizerRuleRegistry`]. Rules are
/// grouped by root expression ID and run in registration order. Nodes are rewritten before their
/// children and again after changed children are installed. A global rewrite budget terminates
/// cyclic rule sets.
#[derive(Debug)]
pub struct BoundExpressionOptimizer {
    registry: OptimizerRuleRegistry,
    max_rewrites: usize,
}

impl Default for BoundExpressionOptimizer {
    fn default() -> Self {
        Self::new(OptimizerRuleRegistry::default())
    }
}

impl BoundExpressionOptimizer {
    /// Create an optimizer from a configured rule registry.
    pub fn new(registry: OptimizerRuleRegistry) -> Self {
        Self {
            registry,
            max_rewrites: DEFAULT_MAX_REWRITES,
        }
    }

    /// Set the maximum number of successful rewrites allowed in one optimization.
    pub fn with_max_rewrites(mut self, max_rewrites: usize) -> Self {
        self.max_rewrites = max_rewrites;
        self
    }

    /// Optimize an entire bound expression tree, cloning the input when it remains unchanged.
    pub fn optimize(&self, expr: &BoundExpression) -> VortexResult<BoundExpression> {
        Ok(self.try_optimize(expr)?.unwrap_or_else(|| expr.clone()))
    }

    /// Optimize an entire bound expression tree, returning `None` when no subtree changed.
    pub fn try_optimize(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        let optimized = OptimizationRun::new(&self.registry, self.max_rewrites).run(expr)?;
        let expression = optimized.as_ref().unwrap_or(expr);
        Ok(find_between(expression)?.or(optimized))
    }
}

/// Mutable state for one optimizer invocation.
struct OptimizationRun<'rules> {
    registry: &'rules OptimizerRuleRegistry,
    max_rewrites: usize,
    rewrite_count: usize,
}

impl<'rules> OptimizationRun<'rules> {
    /// Create a run using the given rule registry and rewrite limit.
    fn new(registry: &'rules OptimizerRuleRegistry, max_rewrites: usize) -> Self {
        Self {
            registry,
            max_rewrites,
            rewrite_count: 0,
        }
    }

    /// Optimize `expr`, returning `None` when the complete tree is unchanged.
    fn run(mut self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        self.optimize_subtree(expr, 0)
    }

    /// Apply the first matching rule to `expression`.
    fn try_apply_rule(
        &mut self,
        expression: &BoundExpression,
    ) -> VortexResult<Option<BoundExpression>> {
        let Some(rules) = self.registry.get(expression.id()) else {
            return Ok(None);
        };
        for rule in rules.iter() {
            let Some(replacement) = rule.rewrite(expression)? else {
                continue;
            };
            let rule_name = rule.name();

            vortex_ensure!(
                replacement.dtype() == expression.dtype(),
                "bound-expression rewrite rule {rule_name} changed dtype from {} to {}",
                expression.dtype(),
                replacement.dtype()
            );
            vortex_ensure!(
                replacement != *expression,
                "bound-expression rewrite rule {rule_name} returned an unchanged expression"
            );
            if self.rewrite_count >= self.max_rewrites {
                vortex_bail!(
                    "Exceeded bound-expression rewrite limit of {} while applying {rule_name} \
                     (possible rewrite cycle)",
                    self.max_rewrites
                );
            }
            self.rewrite_count += 1;
            return Ok(Some(replacement));
        }
        Ok(None)
    }

    /// Optimize a subtree to convergence using copy-on-write rebuilding.
    fn optimize_subtree(
        &mut self,
        original: &BoundExpression,
        depth: usize,
    ) -> VortexResult<Option<BoundExpression>> {
        if depth >= MAX_DEPTH {
            vortex_bail!(
                "Exceeded bound-expression optimization depth limit of \
                 {MAX_DEPTH}"
            );
        }

        let mut current = None;
        loop {
            loop {
                let expression = current.as_ref().unwrap_or(original);
                let Some(replacement) = self.try_apply_rule(expression)? else {
                    break;
                };
                current = Some(replacement);
            }

            let expression = current.as_ref().unwrap_or(original);
            let Some(children) = self.optimize_children(expression, depth)? else {
                return Ok(current);
            };

            let original_dtype = expression.dtype().clone();
            let expression = current.take().unwrap_or_else(|| original.clone());
            let rebuilt = expression.with_children(children)?;
            vortex_ensure!(
                rebuilt.dtype() == &original_dtype,
                "optimizing children changed a node dtype from {original_dtype} to {}",
                rebuilt.dtype()
            );

            let Some(replacement) = self.try_apply_rule(&rebuilt)? else {
                return Ok(Some(rebuilt));
            };
            current = Some(replacement);
        }
    }

    /// Optimize a node's children, allocating a replacement vector only after one changes.
    fn optimize_children(
        &mut self,
        expression: &BoundExpression,
        depth: usize,
    ) -> VortexResult<Option<Vec<BoundExpression>>> {
        let children = expression.children();
        let mut optimized_children = None;

        for (index, child) in children.iter().enumerate() {
            match self.optimize_subtree(child, depth + 1)? {
                Some(optimized_child) => {
                    optimized_children
                        .get_or_insert_with(|| {
                            let mut optimized = Vec::with_capacity(children.len());
                            optimized.extend_from_slice(&children[..index]);
                            optimized
                        })
                        .push(optimized_child);
                }
                None => {
                    if let Some(optimized) = &mut optimized_children {
                        optimized.push(child.clone());
                    }
                }
            }
        }

        Ok(optimized_children)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::BoundExpressionOptimizer;
    use super::OptimizerRule;
    use super::OptimizerRuleRegistry;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::BoundExpression;
    use crate::expr::ExpressionId;
    use crate::expr::bound;
    use crate::scalar::Scalar;
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::is_null::IsNull;
    use crate::scalar_fn::fns::literal::Literal;

    #[derive(Debug)]
    struct IsNullTo(bool);

    impl OptimizerRule for IsNullTo {
        fn expression_id(&self) -> ExpressionId {
            IsNull.id()
        }

        fn rewrite(&self, _expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
            Ok(Some(bound::lit(self.0)))
        }
    }

    #[test]
    fn rules_run_in_registration_order() -> VortexResult<()> {
        let input = bound::is_null(bound::lit(1i32));
        let mut registry = OptimizerRuleRegistry::empty();
        registry.register(IsNullTo(true));
        registry.register(IsNullTo(false));
        let optimizer = BoundExpressionOptimizer::new(registry);

        assert_eq!(optimizer.optimize(&input)?, bound::lit(true));
        Ok(())
    }

    #[derive(Debug)]
    struct IsNullToReducibleTree;

    impl OptimizerRule for IsNullToReducibleTree {
        fn expression_id(&self) -> ExpressionId {
            IsNull.id()
        }

        fn rewrite(&self, _expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
            Ok(Some(bound::and(bound::lit(false), bound::lit(true))))
        }
    }

    #[derive(Debug)]
    struct AndFalse;

    impl OptimizerRule for AndFalse {
        fn expression_id(&self) -> ExpressionId {
            Binary.id()
        }

        fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
            Ok(
                (expr.child(0).as_opt::<Literal>() == Some(&Scalar::from(false)))
                    .then(|| bound::lit(false)),
            )
        }
    }

    #[test]
    fn optimizes_subtrees_introduced_by_rules() -> VortexResult<()> {
        let input = bound::is_null(bound::lit(1i32));
        let mut registry = OptimizerRuleRegistry::empty();
        registry.register(IsNullToReducibleTree);
        registry.register(AndFalse);
        let optimizer = BoundExpressionOptimizer::new(registry);

        assert_eq!(optimizer.optimize(&input)?, bound::lit(false));
        Ok(())
    }

    #[test]
    fn retries_root_rules_after_optimizing_children() -> VortexResult<()> {
        let input = bound::and(bound::is_null(bound::lit(1i32)), bound::lit(true));
        let mut registry = OptimizerRuleRegistry::empty();
        registry.register(IsNullTo(false));
        registry.register(AndFalse);
        let optimizer = BoundExpressionOptimizer::new(registry);

        assert_eq!(optimizer.optimize(&input)?, bound::lit(false));
        Ok(())
    }

    #[derive(Debug)]
    struct WrongDType;

    impl OptimizerRule for WrongDType {
        fn expression_id(&self) -> ExpressionId {
            IsNull.id()
        }

        fn rewrite(&self, _expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
            Ok(Some(bound::lit(1i32)))
        }
    }

    #[test]
    fn rejects_dtype_changes() {
        let input = bound::is_null(bound::lit(1i32));
        let mut registry = OptimizerRuleRegistry::empty();
        registry.register(WrongDType);
        let optimizer = BoundExpressionOptimizer::new(registry);

        assert!(optimizer.optimize(&input).is_err());
    }

    #[derive(Debug)]
    struct Unchanged;

    impl OptimizerRule for Unchanged {
        fn expression_id(&self) -> ExpressionId {
            IsNull.id()
        }

        fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
            Ok(Some(expr.clone()))
        }
    }

    #[test]
    fn rejects_unchanged_replacements() {
        let input = bound::is_null(bound::lit(1i32));
        let mut registry = OptimizerRuleRegistry::empty();
        registry.register(Unchanged);
        let optimizer = BoundExpressionOptimizer::new(registry);

        assert!(optimizer.optimize(&input).is_err());
    }

    #[derive(Debug)]
    struct ToggleBoolean;

    impl OptimizerRule for ToggleBoolean {
        fn expression_id(&self) -> ExpressionId {
            Literal.id()
        }

        fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
            let Some(value) = expr
                .as_opt::<Literal>()
                .and_then(|scalar| scalar.as_bool_opt())
            else {
                return Ok(None);
            };
            Ok(value.value().map(|value| bound::lit(!value)))
        }
    }

    #[test]
    fn rewrite_budget_terminates_cycles() {
        let mut registry = OptimizerRuleRegistry::empty();
        registry.register(ToggleBoolean);
        let optimizer = BoundExpressionOptimizer::new(registry).with_max_rewrites(4);

        assert!(optimizer.optimize(&bound::lit(true)).is_err());
    }

    #[derive(Debug)]
    struct RootToOne(ExpressionId);

    impl OptimizerRule for RootToOne {
        fn expression_id(&self) -> ExpressionId {
            self.0
        }

        fn rewrite(&self, _expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
            Ok(Some(bound::lit(1i32)))
        }
    }

    #[test]
    fn rules_can_target_non_scalar_expression_nodes() -> VortexResult<()> {
        let input = bound::root(DType::Primitive(PType::I32, Nullability::NonNullable));
        let mut registry = OptimizerRuleRegistry::empty();
        registry.register(RootToOne(input.id()));
        let optimizer = BoundExpressionOptimizer::new(registry);

        assert_eq!(optimizer.optimize(&input)?, bound::lit(1i32));
        Ok(())
    }

    #[test]
    fn depth_limit_prevents_stack_overflow() {
        let mut expr = bound::root(DType::Bool(Nullability::NonNullable));
        for _ in 0..1_000 {
            expr = bound::not(expr);
        }

        assert!(
            BoundExpressionOptimizer::new(OptimizerRuleRegistry::empty())
                .try_optimize(&expr)
                .is_err()
        );
    }

    #[test]
    fn root_rewrite_can_discard_a_deep_subtree() -> VortexResult<()> {
        let mut discarded = bound::root(DType::Bool(Nullability::NonNullable));
        for _ in 0..1_000 {
            discarded = bound::not(discarded);
        }
        let input = bound::and(bound::lit(false), discarded);
        let mut registry = OptimizerRuleRegistry::empty();
        registry.register(AndFalse);
        let optimizer = BoundExpressionOptimizer::new(registry);

        assert_eq!(optimizer.optimize(&input)?, bound::lit(false));
        Ok(())
    }

    #[test]
    fn default_optimizer_folds_literal_cast() -> VortexResult<()> {
        let target = DType::Primitive(PType::I64, Nullability::NonNullable);
        let expr = bound::cast(bound::lit(1i32), target);

        assert_eq!(
            BoundExpressionOptimizer::default().optimize(&expr)?,
            bound::lit(1i64)
        );
        Ok(())
    }
}

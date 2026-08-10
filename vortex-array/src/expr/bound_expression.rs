// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::display::DisplayTreeExpr;
use crate::expr::lambda::Lambda;
use crate::expr::scope::Frame;
use crate::expr::scope::Scope;
use crate::expr::variable::Variable;
use crate::scalar_fn::ScalarFnRef;

/// An [`Expression`] that has been type-checked against a [`Scope`].
///
/// Every node carries its own dtype, so reading one is a field access rather than a walk of the
/// subtree. Holding a `BoundExpression` is proof that the whole tree type-checked.
///
/// Binding is purely logical: it deals only in [`DType`]s and never sees an array, a length, or an
/// encoding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundExpression {
    /// A scalar function applied to bound children.
    Scalar {
        /// The dtype this node evaluates to.
        dtype: DType,
        /// The scalar function for this node.
        scalar_fn: ScalarFnRef,
        /// The bound children, in argument order.
        ///
        /// Sharing keeps clones cheap even though the iterative [`Drop`] implementation prevents
        /// consumers from destructuring a `BoundExpression` by value.
        children: Arc<Vec<BoundExpression>>,
    },
    /// The scope itself. Its dtype is the scope's root dtype.
    Root {
        /// The dtype this node evaluates to.
        dtype: DType,
    },
    /// A resolved reference to a bound variable.
    Variable {
        /// The dtype this node evaluates to.
        dtype: DType,
        /// The variable that was resolved.
        variable: Variable,
        /// Index of the frame it resolved in, counted from the outermost. Kept so a capture check
        /// can compare it against the depth at a binder without redoing resolution.
        depth: usize,
    },
    /// A lambda.
    ///
    /// The only variant without a dtype: a lambda is not a value. Its function type is recorded
    /// structurally on [`BoundLambda`] instead — `param_dtypes` is the argument side and
    /// [`body_dtype`](BoundLambda::body_dtype) the result side.
    Lambda(BoundLambda),
}

/// A bound lambda.
///
/// A struct as well as a variant, mirroring [`Expression::Lambda`]: the struct lets an API that
/// wants a lambda — a higher-order function — demand one in its signature, while the variant keeps
/// a node for traversal to see.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundLambda {
    /// The parameters and their dtypes: the argument side of the function type.
    ///
    /// Storing the [`Frame`] the body was bound under, rather than two parallel arrays, makes the
    /// name/dtype pairing and the duplicate-name rejection structural instead of maintained by
    /// hand.
    frame: Frame,
    body: Arc<BoundExpression>,
}

impl BoundLambda {
    /// The frame this lambda's body was bound under.
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    /// The parameters paired with their dtypes, in declaration order.
    pub fn bindings(&self) -> &[(Variable, DType)] {
        self.frame.bindings()
    }

    /// The variables this lambda binds, in declaration order.
    ///
    /// An iterator rather than a slice, because the names and dtypes are interleaved in the frame.
    pub fn params(&self) -> impl ExactSizeIterator<Item = &Variable> {
        self.frame.bindings().iter().map(|(variable, _)| variable)
    }

    /// The dtypes of the parameters, in declaration order.
    pub fn param_dtypes(&self) -> impl ExactSizeIterator<Item = &DType> {
        self.frame.bindings().iter().map(|(_, dtype)| dtype)
    }

    /// The bound body.
    pub fn body(&self) -> &BoundExpression {
        &self.body
    }

    /// Return this lambda with a different body, keeping its parameters and their dtypes.
    pub fn with_body(&self, body: BoundExpression) -> Self {
        Self {
            frame: self.frame.clone(),
            body: Arc::new(body),
        }
    }

    /// The dtype the body evaluates to — the result side of the function type.
    pub fn body_dtype(&self) -> Option<&DType> {
        self.body.dtype()
    }
}

impl From<BoundLambda> for BoundExpression {
    fn from(lambda: BoundLambda) -> Self {
        BoundExpression::Lambda(lambda)
    }
}

/// A bound-expression wrapper that compares shared tree identity instead of structure.
#[derive(Clone, Debug)]
pub struct ExactBoundExpr(pub BoundExpression);

impl PartialEq for ExactBoundExpr {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (
                BoundExpression::Root { dtype: lhs_dtype },
                BoundExpression::Root { dtype: rhs_dtype },
            ) => lhs_dtype == rhs_dtype,
            (
                BoundExpression::Scalar {
                    dtype: lhs_dtype,
                    scalar_fn: lhs_fn,
                    children: lhs_children,
                },
                BoundExpression::Scalar {
                    dtype: rhs_dtype,
                    scalar_fn: rhs_fn,
                    children: rhs_children,
                },
            ) => {
                lhs_fn == rhs_fn
                    && Arc::ptr_eq(lhs_children, rhs_children)
                    && lhs_dtype == rhs_dtype
            }
            // No catch-all: a new variant must state its own identity rather than silently
            // comparing unequal, which would put `eq` out of step with `hash`.
            (
                BoundExpression::Variable {
                    dtype: lhs_dtype,
                    variable: lhs_var,
                    depth: lhs_depth,
                },
                BoundExpression::Variable {
                    dtype: rhs_dtype,
                    variable: rhs_var,
                    depth: rhs_depth,
                },
            ) => lhs_var == rhs_var && lhs_depth == rhs_depth && lhs_dtype == rhs_dtype,
            (BoundExpression::Lambda(lhs), BoundExpression::Lambda(rhs)) => {
                lhs.frame == rhs.frame && Arc::ptr_eq(&lhs.body, &rhs.body)
            }
            // No catch-all: a new variant must state its own identity, or `eq` drifts out of step
            // with `hash` and keys stop equalling themselves.
            (BoundExpression::Root { .. }, _)
            | (BoundExpression::Scalar { .. }, _)
            | (BoundExpression::Variable { .. }, _)
            | (BoundExpression::Lambda(_), _) => false,
        }
    }
}

impl Eq for ExactBoundExpr {}

impl Hash for ExactBoundExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // DType differences are resolved by equality. Omitting the potentially lazy dtype keeps
        // identity-keyed cache lookups from deserializing an entire schema just to compute a hash.
        match &self.0 {
            BoundExpression::Root { .. } => state.write_u8(0),
            BoundExpression::Variable {
                variable, depth, ..
            } => {
                state.write_u8(2);
                variable.hash(state);
                depth.hash(state);
            }
            BoundExpression::Lambda(lambda) => {
                state.write_u8(3);
                lambda.frame.hash(state);
                Arc::as_ptr(&lambda.body).hash(state);
            }
            BoundExpression::Scalar {
                scalar_fn,
                children,
                ..
            } => {
                state.write_u8(1);
                scalar_fn.hash(state);
                Arc::as_ptr(children).hash(state);
            }
        }
    }
}

impl BoundExpression {
    /// Create a bound root expression with the given dtype.
    pub fn new_root(dtype: DType) -> Self {
        Self::Root { dtype }
    }

    /// Create a bound scalar node from a scalar function and already-bound children.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: impl IntoIterator<Item = BoundExpression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        vortex_ensure!(
            scalar_fn.signature().arity().matches(children.len()),
            "Expression arity mismatch: expected {} children but got {}",
            scalar_fn.signature().arity(),
            children.len()
        );

        let arg_dtypes = children
            .iter()
            .map(|child| {
                child.dtype().cloned().ok_or_else(|| {
                    vortex_err!("a scalar function argument must be a value, got {child}")
                })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let dtype = scalar_fn.return_dtype(&arg_dtypes)?;

        Ok(Self::Scalar {
            dtype,
            scalar_fn,
            children: children.into(),
        })
    }

    /// Rebuild this node with new bound children, recomputing its dtype.
    pub fn with_children(
        self,
        children: impl IntoIterator<Item = BoundExpression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        match &self {
            BoundExpression::Scalar { scalar_fn, .. } => Self::try_new(scalar_fn.clone(), children),
            BoundExpression::Lambda(lambda) => {
                vortex_ensure!(
                    children.len() == 1,
                    "a lambda expects 1 child but got {}",
                    children.len()
                );
                let body = children
                    .into_iter()
                    .next()
                    .vortex_expect("length checked above");
                Ok(BoundExpression::Lambda(lambda.with_body(body)))
            }
            BoundExpression::Root { .. } | BoundExpression::Variable { .. } => {
                vortex_ensure!(
                    children.is_empty(),
                    "{self} cannot have {} children",
                    children.len()
                );
                Ok(self)
            }
        }
    }

    /// The dtype this expression evaluates to.
    /// The dtype this expression evaluates to, erroring for a lambda, which is not a value.
    ///
    /// Prefer this where the node is a value by construction; it turns what would otherwise be an
    /// `unwrap` at each call site into one clear error.
    pub fn value_dtype(&self) -> VortexResult<&DType> {
        self.dtype()
            .ok_or_else(|| vortex_err!("expected a value, got a lambda: {self}"))
    }

    /// The dtype this expression evaluates to, or `None` for a lambda, which is not a value.
    pub fn dtype(&self) -> Option<&DType> {
        match self {
            Self::Scalar { dtype, .. } | Self::Root { dtype } | Self::Variable { dtype, .. } => {
                Some(dtype)
            }
            Self::Lambda(_) => None,
        }
    }

    /// The bound children of this node, in argument order. Empty for [`BoundExpression::Root`].
    pub fn children(&self) -> &[BoundExpression] {
        match self {
            Self::Scalar { children, .. } => children.as_slice(),
            Self::Lambda(lambda) => std::slice::from_ref(&lambda.body),
            Self::Root { .. } | Self::Variable { .. } => &[],
        }
    }

    /// The scalar function for this node, or `None` if it is the scope root.
    pub fn as_scalar(&self) -> Option<&ScalarFnRef> {
        match self {
            Self::Scalar { scalar_fn, .. } => Some(scalar_fn),
            Self::Root { .. } | Self::Variable { .. } | Self::Lambda(_) => None,
        }
    }

    /// The variable and its resolution depth, if this node is a variable reference.
    pub fn as_variable(&self) -> Option<(&Variable, usize)> {
        match self {
            Self::Variable {
                variable, depth, ..
            } => Some((variable, *depth)),
            Self::Scalar { .. } | Self::Root { .. } | Self::Lambda(_) => None,
        }
    }

    /// Whether this node is the scope root.
    pub fn is_root(&self) -> bool {
        matches!(self, Self::Root { .. })
    }

    /// Display the bound expression as a formatted tree structure.
    pub fn display_tree(&self) -> impl Display {
        DisplayTreeExpr(self)
    }

    /// Convert this bound tree back into its unbound logical representation.
    ///
    /// This rebuilds the expression iteratively; the bound representation does not retain a
    /// second expression tree.
    // TODO: This is temporary artifact of the migration from using `Expression`s to
    // `BoundExpression`s
    pub fn unbind(&self) -> Expression {
        let mut pending = vec![(self, false)];
        let mut expressions = Vec::new();

        while let Some((node, visited)) = pending.pop() {
            match node {
                BoundExpression::Root { .. } => expressions.push(crate::expr::root()),
                BoundExpression::Variable { variable, .. } => {
                    expressions.push(Expression::Variable(variable.clone()))
                }
                BoundExpression::Lambda(lambda) if visited => {
                    let body = expressions.pop().vortex_expect("body was pushed");
                    expressions.push(Expression::from(Lambda::new(
                        lambda.params().cloned(),
                        body,
                    )));
                }
                BoundExpression::Lambda(lambda) => {
                    pending.push((node, true));
                    pending.push((&lambda.body, false));
                }
                BoundExpression::Scalar {
                    scalar_fn,
                    children,
                    ..
                } if visited => {
                    let child_start = expressions.len() - children.len();
                    let child_expressions = expressions.split_off(child_start);
                    expressions.push(
                        Expression::try_new(scalar_fn.clone(), child_expressions)
                            .vortex_expect("a bound expression always has valid arity"),
                    );
                }
                BoundExpression::Scalar { children, .. } => {
                    pending.push((node, true));
                    pending.extend(children.iter().rev().map(|child| (child, false)));
                }
            }
        }

        expressions
            .pop()
            .vortex_expect("binding always produces one expression root")
    }
}

impl Display for BoundExpression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar { scalar_fn, .. } => scalar_fn.fmt_sql(self, f),
            Self::Root { .. } => f.write_str("$"),
            Self::Variable { variable, .. } => write!(f, "${variable}"),
            Self::Lambda(lambda) => {
                write!(f, "({}) -> {}", lambda.params().join(", "), lambda.body)
            }
        }
    }
}

impl Expression {
    /// Bind this expression against a root dtype, type-checking every node in a single walk.
    ///
    /// The returned tree carries a dtype on each node, so callers needing types at more than one
    /// node should bind once and read fields rather than calling
    /// [`return_dtype`](Expression::return_dtype) repeatedly.
    pub fn bind(&self, dtype: &DType) -> VortexResult<BoundExpression> {
        self.bind_scope(&Scope::new(dtype.clone()))
    }

    /// Bind this expression against an explicit [`Scope`].
    ///
    /// Errors on a lambda: a lambda has no dtype, so only whoever supplies its parameter types can
    /// bind it. See [`Lambda::bind`].
    pub fn bind_scope(&self, scope: &Scope) -> VortexResult<BoundExpression> {
        match self {
            Expression::Root => Ok(BoundExpression::new_root(scope.root().clone())),
            Expression::Variable(variable) => {
                let (dtype, depth) = scope.resolve(variable).ok_or_else(|| {
                    vortex_err!(
                        "unbound variable '{variable}'; the scope binds {} frame(s)",
                        scope.depth()
                    )
                })?;
                Ok(BoundExpression::Variable {
                    dtype: dtype.clone(),
                    variable: variable.clone(),
                    depth,
                })
            }
            Expression::Lambda(lambda) => vortex_bail!(
                "a lambda ({}) is not a value and cannot be bound here; use Lambda::bind, \
                 which supplies its parameter types",
                lambda.params().iter().join(", ")
            ),
            Expression::Scalar {
                scalar_fn,
                children,
            } => {
                let children: Vec<_> = children
                    .iter()
                    .map(|child| child.bind_scope(scope))
                    .try_collect()?;
                BoundExpression::try_new(scalar_fn.clone(), children)
            }
        }
    }
}

impl Lambda {
    /// Bind this lambda, supplying the dtypes of its parameters.
    ///
    /// The parameter dtypes come from the caller because a lambda cannot know them — they are
    /// determined by whatever applies it. A higher-order function derives them from its own
    /// arguments and calls this.
    ///
    /// The body is bound under `scope` extended with one frame holding the parameters, so a
    /// parameter shadows an outer binding of the same name.
    pub fn bind(
        &self,
        scope: &Scope,
        param_dtypes: impl IntoIterator<Item = DType>,
    ) -> VortexResult<BoundLambda> {
        let param_dtypes = Vec::from_iter(param_dtypes);
        vortex_ensure!(
            param_dtypes.len() == self.params().len(),
            "lambda binds {} parameter(s) but {} dtype(s) were supplied",
            self.params().len(),
            param_dtypes.len()
        );

        let frame = Frame::try_new(
            self.params()
                .iter()
                .cloned()
                .zip(param_dtypes.iter().cloned()),
        )?;
        let body = self.body().bind_scope(&scope.push_frame(frame.clone()))?;

        Ok(BoundLambda {
            frame,
            body: Arc::new(body),
        })
    }
}

/// Iterative drop to avoid stack overflows on deep trees.
impl Drop for BoundExpression {
    fn drop(&mut self) {
        let Self::Scalar { children, .. } = self else {
            return;
        };
        let Some(children) = Arc::get_mut(children) else {
            return;
        };

        let mut to_drop = std::mem::take(children);
        while let Some(mut child) = to_drop.pop() {
            if let BoundExpression::Scalar { children, .. } = &mut child
                && let Some(grandchildren) = Arc::get_mut(children)
            {
                to_drop.append(grandchildren);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::expr::test_harness::struct_dtype;

    fn scope() -> Scope {
        Scope::new(struct_dtype())
    }

    #[test]
    fn root_binds_to_the_scope() -> VortexResult<()> {
        let bound = root().bind_scope(&scope())?;
        assert!(bound.is_root());
        assert_eq!(bound.dtype(), Some(&struct_dtype()));
        assert_eq!(bound.unbind(), root());
        Ok(())
    }

    #[test]
    fn every_node_carries_its_dtype() -> VortexResult<()> {
        let expr = eq(col("a"), lit(1_i32));
        let bound = expr.bind_scope(&scope())?;

        assert_eq!(bound.dtype(), Some(&DType::Bool(Nullability::NonNullable)));

        let lhs = &bound.children()[0];
        assert_eq!(
            lhs.dtype(),
            Some(&DType::Primitive(PType::I32, Nullability::NonNullable))
        );
        assert_eq!(lhs.children()[0].dtype(), Some(&struct_dtype()));
        Ok(())
    }

    #[test]
    fn bind_agrees_with_return_dtype() -> VortexResult<()> {
        for expr in [root(), col("a"), eq(col("a"), lit(1_i32)), lit(true)] {
            assert_eq!(
                expr.bind(&struct_dtype())?.dtype(),
                Some(&expr.return_dtype(struct_dtype())?),
                "disagreement for {expr}"
            );
        }
        Ok(())
    }

    #[test]
    fn bound_display_matches_unbound() -> VortexResult<()> {
        for expr in [root(), col("a"), eq(col("a"), lit(1_i32)), lit(true)] {
            let bound = expr.bind_scope(&scope())?;
            assert_eq!(bound.to_string(), expr.to_string());
            assert_eq!(
                bound.display_tree().to_string(),
                expr.display_tree().to_string()
            );
        }
        Ok(())
    }

    #[test]
    fn clone_shares_children() -> VortexResult<()> {
        let bound = eq(col("a"), lit(1_i32)).bind_scope(&scope())?;
        let cloned = bound.clone();

        let (
            BoundExpression::Scalar { children: a, .. },
            BoundExpression::Scalar { children: b, .. },
        ) = (&bound, &cloned)
        else {
            unreachable!("eq is a scalar node")
        };
        assert!(Arc::ptr_eq(a, b));
        Ok(())
    }

    #[test]
    fn repeated_subtree_is_bound_per_occurrence() -> VortexResult<()> {
        let shared = col("a");
        let bound = eq(shared.clone(), shared).bind_scope(&scope())?;
        let children = bound.children();
        assert_eq!(children[0].dtype(), children[1].dtype());
        Ok(())
    }

    #[test]
    fn structural_and_exact_equality_are_distinct() -> VortexResult<()> {
        let expr = eq(col("a"), lit(1_i32));
        let bound = expr.bind_scope(&scope())?;
        let independently_bound = expr.bind_scope(&scope())?;

        assert_eq!(bound, independently_bound);
        assert_eq!(ExactBoundExpr(bound.clone()), ExactBoundExpr(bound.clone()));
        assert_ne!(
            ExactBoundExpr(bound.clone()),
            ExactBoundExpr(independently_bound)
        );
        assert_eq!(bound.unbind(), expr);
        Ok(())
    }

    #[test]
    fn binding_reports_a_type_error() {
        let expr = eq(col("a"), lit("nope"));
        assert!(expr.bind_scope(&scope()).is_err());
    }
}

#[cfg(test)]
mod lambda_tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::checked_add;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::lambda;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::expr::test_harness::struct_dtype;
    use crate::expr::var;

    fn i32_() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    fn scope() -> Scope {
        Scope::new(struct_dtype())
    }

    #[test]
    fn a_lambda_body_sees_its_parameters() -> VortexResult<()> {
        let l = lambda(["x"], checked_add(var("x"), lit(1i32)));
        let bound = l.bind(&scope(), [i32_()])?;

        assert_eq!(
            bound.param_dtypes().cloned().collect::<Vec<_>>(),
            vec![i32_()]
        );
        assert_eq!(bound.body_dtype(), Some(&i32_()));
        Ok(())
    }

    /// A lambda has no dtype, so it cannot be bound as a value.
    #[test]
    fn binding_a_bare_lambda_fails() {
        let l = lambda(["x"], var("x"));
        assert!(Expression::from(l).bind_scope(&scope()).is_err());
    }

    /// The same reason: a lambda in an argument position is not a value.
    #[test]
    fn a_lambda_in_a_value_position_fails() {
        let expr = eq(Expression::from(lambda(["x"], var("x"))), lit(1i32));
        assert!(expr.bind_scope(&scope()).is_err());
    }

    #[test]
    fn an_unbound_variable_fails() {
        assert!(var("nope").bind_scope(&scope()).is_err());
        // ...and also when it is merely the wrong name inside a lambda.
        let l = lambda(["x"], var("y"));
        assert!(l.bind(&scope(), [i32_()]).is_err());
    }

    #[test]
    fn parameter_count_must_match_supplied_dtypes() {
        let l = lambda(["x", "y"], var("x"));
        assert!(l.bind(&scope(), [i32_()]).is_err());
    }

    #[test]
    fn duplicate_parameters_are_rejected() {
        let l = lambda(["x", "x"], var("x"));
        assert!(l.bind(&scope(), [i32_(), i32_()]).is_err());
    }

    /// The body binds under a pushed frame, so a parameter shadows an outer binding, and the
    /// recorded depth distinguishes the two.
    #[test]
    fn an_inner_parameter_shadows_an_outer_binding() -> VortexResult<()> {
        let outer = scope().push_frame(Frame::try_new([(
            Variable::new("x"),
            DType::Utf8(Nullability::NonNullable),
        )])?);

        let bound = lambda(["x"], var("x")).bind(&outer, [i32_()])?;
        assert_eq!(
            bound.body_dtype(),
            Some(&i32_()),
            "inner parameter should win"
        );

        let (_, depth) = bound
            .body()
            .as_variable()
            .vortex_expect("body is a variable");
        assert_eq!(
            depth, 1,
            "resolved in the lambda's own frame, not the outer"
        );
        Ok(())
    }

    /// Root still resolves inside a lambda body: it is a capture, which nothing rejects yet.
    #[test]
    fn root_is_still_reachable_from_a_body() -> VortexResult<()> {
        let bound = lambda(["x"], col("a")).bind(&scope(), [i32_()])?;
        assert_eq!(bound.body_dtype(), Some(&i32_()));
        Ok(())
    }

    #[test]
    fn a_variable_round_trips_through_unbind() -> VortexResult<()> {
        let bound = lambda(["x"], var("x")).bind(&scope(), [i32_()])?;
        assert_eq!(bound.body().unbind(), var("x"));
        Ok(())
    }

    /// `ExactBoundExpr` is a `HashMap` key, so `eq` and `hash` must agree. If `eq` falls through
    /// to a catch-all for a variant that `hash` distinguishes, a key stops being equal to itself
    /// and lookups miss.
    #[test]
    fn a_bound_variable_is_a_usable_map_key() -> VortexResult<()> {
        use std::collections::hash_map::RandomState;
        use std::hash::BuildHasher;

        let bound = lambda(["x"], var("x")).bind(&scope(), [i32_()])?;
        let key = ExactBoundExpr(bound.body().clone());
        let same = key.clone();

        // `ExactBoundExpr` is a map key, so `eq` and `hash` must agree. A catch-all `eq` arm made
        // a variable unequal to itself while `hash` still distinguished it, so lookups missed.
        assert_eq!(key, same, "a variable key must equal itself");

        let state = RandomState::new();
        assert_eq!(state.hash_one(&key), state.hash_one(&same));
        Ok(())
    }

    /// Execution has no variable environment, so a bound variable must error rather than silently
    /// evaluating as the scope.
    #[test]
    fn applying_a_bound_variable_errors() -> VortexResult<()> {
        use crate::IntoArray;
        use crate::arrays::PrimitiveArray;

        let array = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let bound = lambda(["x"], var("x")).bind(&scope(), [i32_()])?;

        assert!(array.apply_bound(bound.body()).is_err());
        Ok(())
    }

    /// The unbound path must error too, rather than panicking on the missing scalar fn.
    #[test]
    fn applying_an_unbound_variable_or_lambda_errors() {
        use crate::IntoArray;
        use crate::arrays::PrimitiveArray;

        let array = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        assert!(array.clone().apply(&var("x")).is_err());
        assert!(
            array
                .apply(&Expression::from(lambda(["x"], var("x"))))
                .is_err()
        );
    }

    /// A deep lambda chain must drop iteratively; recursing through bodies overflows the stack.
    #[test]
    fn a_deep_lambda_chain_drops_without_overflowing() {
        let mut expr = var("x");
        for _ in 0..100_000 {
            expr = Expression::from(lambda(["x"], expr));
        }
        drop(expr);
    }

    /// `iter_children` and `children_count` count a lambda's body as a child, so `apply_children`
    /// and `map_children` must visit it too. When they did not, `LabelingVisitor::visit_up` folded
    /// over a child that had never been labelled and panicked.
    #[test]
    fn labelling_visits_a_bound_lambda_body() -> VortexResult<()> {
        use crate::expr::analysis::label_tree;
        use crate::expr::traversal::Node;

        let bound = BoundExpression::from(lambda(["x"], var("x")).bind(&scope(), [i32_()])?);
        assert_eq!(bound.children_count(), 1, "a lambda's body is a child");

        // Counts nodes, so an unvisited child panics rather than miscounting.
        let labels = label_tree(&bound, |_| 1usize, |acc, child| acc + child);
        assert_eq!(labels.get(&bound), Some(&2), "the lambda and its body");
        Ok(())
    }

    /// Rewriting must reach the body too, and rebuild the lambda around the result.
    #[test]
    fn rewriting_reaches_a_bound_lambda_body() -> VortexResult<()> {
        use crate::expr::traversal::NodeExt;
        use crate::expr::traversal::Transformed;

        let bound = BoundExpression::from(lambda(["x"], var("x")).bind(&scope(), [i32_()])?);

        let mut visited = 0;
        let rewritten = bound.clone().transform_up(|node| {
            visited += 1;
            Ok(Transformed::no(node))
        })?;
        assert_eq!(visited, 2, "the lambda and its body");
        assert_eq!(rewritten.into_inner(), bound);
        Ok(())
    }

    /// A bound lambda stores the frame its body was bound under, so the name/dtype pairing cannot
    /// drift the way two parallel arrays could.
    #[test]
    fn a_bound_lambda_keeps_the_frame_it_bound_under() -> VortexResult<()> {
        let bound = lambda(["x"], var("x")).bind(&scope(), [i32_()])?;

        assert_eq!(
            bound.params().collect::<Vec<_>>(),
            vec![&Variable::new("x")]
        );
        assert_eq!(bound.param_dtypes().collect::<Vec<_>>(), vec![&i32_()]);
        assert_eq!(bound.frame().get(&Variable::new("x")), Some(&i32_()));
        Ok(())
    }

    /// Parameters are named, so equality is structural rather than alpha-equivalence: `x -> x`
    /// and `y -> y` denote the same function but compare unequal. Nothing depends on the stronger
    /// property today; pinning it here makes the choice explicit rather than accidental. Switching
    /// to de Bruijn coordinates would make alpha-equivalence structural.
    #[test]
    fn lambdas_are_not_alpha_equivalent() {
        assert_ne!(lambda(["x"], var("x")), lambda(["y"], var("y")));
        assert_eq!(lambda(["x"], var("x")), lambda(["x"], var("x")));
    }

    #[test]
    fn display_shows_the_binder_and_the_reference() {
        assert_eq!(var("x").to_string(), "$x");
        assert_eq!(
            lambda(["x"], checked_add(var("x"), lit(1i32))).to_string(),
            "(x) -> ($x + 1i32)"
        );
        assert_eq!(root().to_string(), "$");
    }
}

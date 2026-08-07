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
    params: Box<[Variable]>,
    param_dtypes: Arc<Vec<DType>>,
    body: Arc<BoundExpression>,
}

impl BoundLambda {
    /// The variables this lambda binds, in declaration order.
    pub fn params(&self) -> &[Variable] {
        &self.params
    }

    /// The dtypes of the parameters, in declaration order.
    pub fn param_dtypes(&self) -> &[DType] {
        &self.param_dtypes
    }

    /// The bound body.
    pub fn body(&self) -> &BoundExpression {
        &self.body
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
                lhs.params == rhs.params
                    && lhs.param_dtypes == rhs.param_dtypes
                    && Arc::ptr_eq(&lhs.body, &rhs.body)
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
                lambda.params.hash(state);
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
        let BoundExpression::Scalar { scalar_fn, .. } = &self else {
            vortex_ensure!(
                children.is_empty(),
                "Root expression cannot have {} children",
                children.len()
            );
            return Ok(self);
        };

        Self::try_new(scalar_fn.clone(), children)
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
                        lambda.params.iter().cloned(),
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
            Self::Lambda(lambda) => write!(
                f,
                "({}) -> {}",
                lambda.params.iter().join(", "),
                lambda.body
            ),
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
        let body = self.body().bind_scope(&scope.push_frame(frame))?;

        Ok(BoundLambda {
            params: self.params().into(),
            param_dtypes: Arc::new(param_dtypes),
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
                Some(&expr.return_dtype(&struct_dtype())?),
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

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
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::Lambda;
use crate::expr::display::DisplayTreeExpr;
use crate::expr::scope::Scope;
use crate::expr::scope::VariableRef;
use crate::expr::traversal::TraversalOrder;
use crate::expr::traversal::pre_order_visit_down;
use crate::expr::variable::Variable;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::stats::rewrite::StatsRewriteCtx;

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
    /// A lambda with a bound body and bound params.
    Lambda(BoundLambda),
    /// The scope itself. Its dtype is the scope's root dtype.
    Root {
        /// The dtype this node evaluates to.
        dtype: DType,
    },
    /// A variable with resolved dtype and reference in the bound scope.
    Variable(BoundVariable),
}

/// A bound variable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundVariable {
    dtype: DType,
    variable: Variable,
    variable_ref: VariableRef,
}

impl BoundVariable {
    /// The dtype of the value bound to this variable.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// The source-level name, retained for display and diagnostics.
    pub fn variable(&self) -> &Variable {
        &self.variable
    }

    /// The lexical location resolved while binding.
    pub fn variable_ref(&self) -> VariableRef {
        self.variable_ref
    }
}

impl Display for BoundVariable {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.variable, f)
    }
}

/// A bound lambda.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundLambda {
    params: Arc<Vec<BoundVariable>>,
    body: Arc<BoundExpression>,
}

impl BoundLambda {
    /// The variables this lambda binds, in declaration order.
    pub fn params(&self) -> &[BoundVariable] {
        &self.params
    }

    /// The bound body.
    pub fn body(&self) -> &BoundExpression {
        &self.body
    }

    /// Take the bound body when this lambda is its sole owner.
    fn take_body(&mut self) -> Option<BoundExpression> {
        Arc::try_unwrap(std::mem::replace(
            &mut self.body,
            Arc::new(BoundExpression::new_root(DType::Null)),
        ))
        .ok()
    }

    /// The dtype the body evaluates to.
    pub fn body_dtype(&self) -> VortexResult<&DType> {
        self.body.dtype()
    }
}

impl Display for BoundLambda {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}) -> {}", self.params.iter().join(", "), self.body)
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
            (BoundExpression::Lambda(lhs), BoundExpression::Lambda(rhs)) => lhs == rhs,
            (BoundExpression::Variable(lhs), BoundExpression::Variable(rhs)) => lhs == rhs,
            (BoundExpression::Root { .. }, _)
            | (BoundExpression::Scalar { .. }, _)
            | (BoundExpression::Lambda(_), _)
            | (BoundExpression::Variable(_), _) => false,
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
            BoundExpression::Variable(variable) => {
                state.write_u8(2);
                variable.variable().hash(state);
                variable.variable_ref().hash(state);
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
            BoundExpression::Lambda(lambda) => {
                state.write_u8(4);
                lambda.hash(state);
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
        Self::try_new_vec(scalar_fn, children.into_iter().collect())
    }

    fn try_new_vec(scalar_fn: ScalarFnRef, children: Vec<BoundExpression>) -> VortexResult<Self> {
        vortex_ensure!(
            scalar_fn.signature().arity().matches(children.len()),
            "Expression arity mismatch: expected {} children but got {}",
            scalar_fn.signature().arity(),
            children.len()
        );
        vortex_ensure!(
            children.iter().all(|child| !child.as_lambda().is_some()),
            "a scalar function cannot take a lambda as an ordinary argument"
        );

        let arg_dtypes: Vec<DType> = children
            .iter()
            .map(|child| child.dtype().cloned())
            .try_collect()?;
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
            BoundExpression::Lambda(_)
            | BoundExpression::Root { .. }
            | BoundExpression::Variable(_) => {
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
    ///
    /// A lambda is callable but not array-valued, so it has no standalone dtype.
    pub fn dtype(&self) -> VortexResult<&DType> {
        match self {
            Self::Scalar { dtype, .. } | Self::Root { dtype } => Ok(dtype),
            Self::Lambda(_) => vortex_bail!("a lambda has no standalone dtype"),
            Self::Variable(variable) => Ok(variable.dtype()),
        }
    }

    /// The bound children of this node, in argument order. Empty for [`BoundExpression::Root`].
    pub fn children(&self) -> &[BoundExpression] {
        match self {
            Self::Scalar { children, .. } => children.as_slice(),
            Self::Lambda(_) | Self::Root { .. } | Self::Variable(_) => &[],
        }
    }

    /// Return the child at `index`.
    pub fn child(&self, index: usize) -> &BoundExpression {
        &self.children()[index]
    }

    /// The scalar function for this node, or `None` if it is the scope root.
    pub fn as_scalar(&self) -> Option<&ScalarFnRef> {
        match self {
            Self::Scalar { scalar_fn, .. } => Some(scalar_fn),
            Self::Lambda(_) | Self::Root { .. } | Self::Variable(_) => None,
        }
    }

    /// Return this node's bound lambda, if it is a lambda binder.
    pub fn as_lambda(&self) -> Option<&BoundLambda> {
        match self {
            Self::Lambda(lambda) => Some(lambda),
            Self::Scalar { .. } | Self::Root { .. } | Self::Variable(_) => None,
        }
    }

    /// Return thjis node's bound variable, if it is a bound variable.
    pub fn as_variable(&self) -> Option<&BoundVariable> {
        match self {
            Self::Variable(variable) => Some(variable),
            Self::Scalar { .. } | Self::Lambda(_) | Self::Root { .. } => None,
        }
    }

    /// Return whether this node uses the given scalar-function vtable.
    pub fn is<V: ScalarFnVTable>(&self) -> bool {
        self.as_scalar().is_some_and(ScalarFnRef::is::<V>)
    }

    /// Return whether this expression tree contains a node using the given scalar-function vtable.
    pub fn contains<V: ScalarFnVTable>(&self) -> VortexResult<bool> {
        let mut contains = false;
        pre_order_visit_down(self, |node| {
            if node.is::<V>() {
                contains = true;
                return Ok(TraversalOrder::Stop);
            }
            Ok(TraversalOrder::Continue)
        })?;
        Ok(contains)
    }

    /// Return the typed scalar-function options when this node uses the given vtable.
    pub fn as_opt<V: ScalarFnVTable>(&self) -> Option<&V::Options> {
        self.as_scalar().and_then(ScalarFnRef::as_opt::<V>)
    }

    /// Return the typed scalar-function options for this node.
    ///
    /// # Panics
    ///
    /// Panics when this node is the scope root or uses a different scalar-function vtable.
    pub fn as_<V: ScalarFnVTable>(&self) -> &V::Options {
        self.as_opt::<V>()
            .vortex_expect("Bound expression options type mismatch")
    }

    /// Whether this node is the scope root.
    pub fn is_root(&self) -> bool {
        matches!(self, Self::Root { .. })
    }

    /// Return whether every scope root in this expression has `dtype`.
    ///
    /// Expressions without a scope root, such as literals, match every dtype.
    pub fn is_root_bound_to(&self, dtype: &DType) -> VortexResult<bool> {
        let mut is_bound_to = true;
        pre_order_visit_down(self, |node| {
            if node.is_root() && node.dtype()? != dtype {
                is_bound_to = false;
                return Ok(TraversalOrder::Stop);
            }
            Ok(TraversalOrder::Continue)
        })?;
        Ok(is_bound_to)
    }

    /// Return an expression that proves this predicate is definitely false from statistics.
    pub fn falsify(&self, session: &VortexSession) -> VortexResult<Option<BoundExpression>> {
        StatsRewriteCtx::new(session).falsify(self)
    }

    /// Return an expression that proves this predicate is definitely true from statistics.
    pub fn satisfy(&self, session: &VortexSession) -> VortexResult<Option<BoundExpression>> {
        StatsRewriteCtx::new(session).satisfy(self)
    }

    /// Display the bound expression as a formatted tree structure.
    pub fn display_tree(&self) -> impl Display {
        DisplayTreeExpr(self)
    }
}

impl Display for BoundExpression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar { scalar_fn, .. } => scalar_fn.fmt_sql(self, f),
            Self::Lambda(lambda) => Display::fmt(lambda, f),
            Self::Root { .. } => f.write_str("$"),
            Self::Variable(variable) => write!(f, "${variable}"),
        }
    }
}

impl Expression {
    /// Bind this expression against `scope`, type-checking every node in a single walk.
    ///
    /// The returned tree carries a dtype on each node, so callers needing types at more than one
    /// node should bind once and read fields rather than calling
    /// [`return_dtype`](Expression::return_dtype) repeatedly.
    pub fn bind(&self, scope: impl Into<Scope>) -> VortexResult<BoundExpression> {
        self.bind_inner(&scope.into())
    }

    fn bind_inner(&self, scope: &Scope) -> VortexResult<BoundExpression> {
        match self {
            Expression::Root => Ok(BoundExpression::new_root(scope.root().clone())),
            Expression::Variable(variable) => {
                let Some((dtype, variable_ref)) = scope.resolve(variable) else {
                    vortex_bail!("unbound variable '{variable}'");
                };
                Ok(BoundExpression::Variable(BoundVariable {
                    dtype: dtype.clone(),
                    variable: variable.clone(),
                    variable_ref,
                }))
            }
            Expression::Scalar {
                scalar_fn,
                children,
            } => {
                let children: Vec<_> = children
                    .iter()
                    .map(|child| child.bind_inner(scope))
                    .try_collect()?;
                BoundExpression::try_new(scalar_fn.clone(), children)
            }
            Expression::Lambda(lambda) => {
                Ok(BoundExpression::Lambda(Self::bind_lambda(lambda, scope)?))
            }
        }
    }

    fn bind_lambda(lambda: &Lambda, scope: &Scope) -> VortexResult<BoundLambda> {
        vortex_ensure!(
            scope.depth() > 0,
            "a lambda can be bound only in a scope with a parameter frame"
        );
        let parameter_frame = scope.depth() - 1;
        let params = lambda
            .params()
            .iter()
            .map(|parameter| {
                scope
                    .resolve(parameter)
                    .map(|(dtype, variable_ref)| BoundVariable {
                        dtype: dtype.clone(),
                        variable: parameter.clone(),
                        variable_ref,
                    })
                    .ok_or_else(|| {
                        vortex_err!("lambda parameter '{parameter}' is not bound in its scope")
                    })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        vortex_ensure!(
            params
                .iter()
                .all(|parameter| parameter.variable_ref().frame() == parameter_frame),
            "lambda parameters must be bound in the innermost lexical frame"
        );

        Ok(BoundLambda {
            params: Arc::new(params),
            body: Arc::new(lambda.body().bind_inner(scope)?),
        })
    }
}

fn drain_bound_drop_children(expression: &mut BoundExpression, to_drop: &mut Vec<BoundExpression>) {
    match expression {
        BoundExpression::Scalar { children, .. } => {
            if let Some(children) = Arc::get_mut(children) {
                to_drop.append(children);
            }
        }
        BoundExpression::Lambda(lambda) => {
            if let Some(body) = lambda.take_body() {
                to_drop.push(body);
            }
        }
        BoundExpression::Root { .. } | BoundExpression::Variable(_) => {}
    }
}

/// Iterative drop to avoid stack overflows on deep value trees and lambda bodies.
impl Drop for BoundExpression {
    fn drop(&mut self) {
        let mut to_drop = Vec::new();
        drain_bound_drop_children(self, &mut to_drop);
        while let Some(mut expression) = to_drop.pop() {
            drain_bound_drop_children(&mut expression, &mut to_drop);
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;
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
    use crate::scalar_fn::fns::is_not_null::IsNotNull;
    use crate::scalar_fn::fns::literal::Literal;

    fn scope() -> Scope {
        Scope::new(struct_dtype())
    }

    #[test]
    fn root_binds_to_the_scope() -> VortexResult<()> {
        let bound = root().bind(scope())?;
        assert!(bound.is_root());
        assert_eq!(bound.dtype()?, &struct_dtype());
        assert_eq!(bound, BoundExpression::new_root(struct_dtype()));
        Ok(())
    }

    #[test]
    fn every_node_carries_its_dtype() -> VortexResult<()> {
        let expr = eq(col("a"), lit(1_i32));
        let bound = expr.bind(scope())?;

        assert_eq!(bound.dtype()?, &DType::Bool(Nullability::NonNullable));

        let lhs = &bound.children()[0];
        assert_eq!(
            lhs.dtype()?,
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(lhs.children()[0].dtype()?, &struct_dtype());
        Ok(())
    }

    #[test]
    fn bind_agrees_with_return_dtype() -> VortexResult<()> {
        for expr in [root(), col("a"), eq(col("a"), lit(1_i32)), lit(true)] {
            assert_eq!(
                expr.bind(struct_dtype())?.dtype()?,
                &expr.return_dtype(&struct_dtype())?,
                "disagreement for {expr}"
            );
        }
        Ok(())
    }

    #[test]
    fn contains_scalar_function() -> VortexResult<()> {
        let bound = eq(col("a"), lit(1_i32)).bind(scope())?;
        assert!(bound.contains::<Literal>()?);
        assert!(!root().bind(scope())?.contains::<Literal>()?);
        Ok(())
    }

    #[test]
    fn bound_to_checks_every_root() -> VortexResult<()> {
        let dtype = struct_dtype();
        let bound = eq(col("a"), col("a")).bind(&dtype)?;
        assert!(bound.is_root_bound_to(&dtype)?);
        assert!(!bound.is_root_bound_to(&DType::Bool(Nullability::NonNullable))?);
        assert!(
            lit(true)
                .bind(&dtype)?
                .is_root_bound_to(&DType::Bool(Nullability::NonNullable))?
        );
        Ok(())
    }

    #[test]
    fn bound_display_matches_unbound() -> VortexResult<()> {
        for expr in [root(), col("a"), eq(col("a"), lit(1_i32)), lit(true)] {
            let bound = expr.bind(scope())?;
            assert_eq!(bound.to_string(), expr.to_string());
            assert_eq!(
                bound.display_tree().to_string(),
                expr.display_tree().to_string()
            );
        }
        Ok(())
    }

    #[test]
    fn variable_binds_to_its_scope() -> VortexResult<()> {
        let scope = scope().with_bindings([(
            Variable::new("value"),
            DType::Primitive(PType::I64, Nullability::Nullable),
        )])?;

        let bound = var("value").bind(&scope)?;
        let variable = bound
            .as_variable()
            .vortex_expect("variable must remain bound");

        assert_eq!(variable.variable(), &Variable::new("value"));
        assert_eq!(
            bound.dtype()?,
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn variable_validity_is_deferred_to_its_bound_array() -> VortexResult<()> {
        let scope = scope().with_bindings([(
            Variable::new("value"),
            DType::Primitive(PType::I32, Nullability::Nullable),
        )])?;
        let expression = checked_add(var("value"), lit(1_i32));

        let validity = expression.validity()?;
        assert!(validity.contains::<IsNotNull>()?);
        assert_eq!(
            validity.bind(&scope)?.dtype()?,
            &DType::Bool(Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn duplicate_lambda_parameters_are_rejected() {
        assert!(lambda(["x", "x"], var("x")).is_err());
    }

    #[test]
    fn lambda_signature_comes_from_its_scope() -> VortexResult<()> {
        let lambda = lambda(["value"], var("value"))?;
        let value_dtype = DType::Primitive(PType::I64, Nullability::Nullable);
        let lambda_scope =
            scope().with_bindings([(Variable::new("value"), value_dtype.clone())])?;

        assert!(lambda.return_dtype(&struct_dtype()).is_err());

        let bound = lambda.bind(&lambda_scope)?;
        assert!(bound.dtype().is_err());
        let bound = bound
            .as_lambda()
            .vortex_expect("a bound lambda expression must contain a bound lambda");
        assert_eq!(bound.body_dtype()?, &value_dtype);
        let parameter = &bound.params()[0];
        assert_eq!(parameter.dtype(), &value_dtype);
        assert_eq!(parameter.variable(), &Variable::new("value"));
        assert_eq!(parameter.variable_ref().frame(), 0);
        assert_eq!(
            bound
                .body()
                .as_variable()
                .vortex_expect("lambda body must resolve its parameter"),
            parameter
        );
        assert!(lambda.bind(scope()).is_err());
        Ok(())
    }

    #[test]
    fn structural_and_exact_equality_are_distinct() -> VortexResult<()> {
        let expr = eq(col("a"), lit(1_i32));
        let bound = expr.bind(scope())?;
        let independently_bound = expr.bind(scope())?;

        assert_eq!(bound, independently_bound);
        assert_eq!(ExactBoundExpr(bound.clone()), ExactBoundExpr(bound.clone()));
        assert_ne!(ExactBoundExpr(bound), ExactBoundExpr(independently_bound));
        Ok(())
    }

    #[test]
    fn binding_reports_a_type_error() {
        assert!(eq(col("a"), lit("nope")).bind(scope()).is_err());
    }
}

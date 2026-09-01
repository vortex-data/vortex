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

use crate::arrays::list_transform::array::output_dtype;
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
    /// A bound lambda whose dtype is the dtype of its body.
    ///
    /// A lambda is not independently executable; only an enclosing higher-order function may
    /// close it over captures and apply it to arguments.
    Lambda(BoundLambda),
    /// A dedicated list transformation. Its ordinary children are the outer list followed by
    /// capture expressions; its lambda body is a lexical boundary owned by `lambda`.
    ListTransform {
        /// The output list dtype.
        dtype: DType,
        /// The typed lambda, including its bound lexical body.
        lambda: BoundLambda,
        /// Slot 0 is the list input and remaining slots are captures.
        children: Arc<Vec<BoundExpression>>,
    },
    /// The scope itself. Its dtype is the scope's root dtype.
    Root {
        /// The dtype this node evaluates to.
        dtype: DType,
    },
    /// A variable resolved to a dtype and stable location in the bound scope.
    Variable(BoundVariable),
}

/// A variable resolved in a lexical scope.
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

    /// The source-level variable name.
    pub fn variable(&self) -> &Variable {
        &self.variable
    }

    /// The variable's stable location in the bound scope.
    pub fn variable_ref(&self) -> VariableRef {
        self.variable_ref
    }
}

impl Display for BoundVariable {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.variable, f)
    }
}

/// A lambda whose parameters and body have been resolved against a lexical [`Scope`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundLambda {
    params: Box<[Variable]>,
    param_dtypes: Box<[DType]>,
    param_refs: Box<[VariableRef]>,
    captures: Box<[BoundVariable]>,
    parameter_frame: usize,
    body: Arc<BoundExpression>,
}

impl BoundLambda {
    /// Bind `lambda` against a scope containing its parameter bindings in the innermost frame.
    ///
    /// The enclosing higher-order function determines the parameter dtypes and constructs this
    /// scope before binding the lambda.
    pub fn bind(lambda: &Lambda, scope: &Scope) -> VortexResult<Self> {
        vortex_ensure!(
            scope.depth() > 0,
            "lambda parameters must be bound in a lexical frame"
        );
        let parameter_frame = scope.depth() - 1;
        let parameter_bindings = lambda
            .params()
            .iter()
            .map(|param| {
                scope
                    .resolve(param)
                    .map(|(dtype, variable_ref)| (dtype.clone(), variable_ref))
                    .ok_or_else(|| {
                        vortex_err!("lambda parameter '{param}' is not bound in its scope")
                    })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        vortex_ensure!(
            parameter_bindings
                .iter()
                .all(|(_, variable_ref)| variable_ref.frame() == parameter_frame),
            "lambda parameters must be bound in the innermost lexical frame"
        );

        let body = lambda.body().bind_scope(scope)?;
        let captures = collect_captures(&body, parameter_frame);

        Ok(Self {
            params: lambda.params().into(),
            param_dtypes: parameter_bindings
                .iter()
                .map(|(dtype, _)| dtype.clone())
                .collect(),
            param_refs: parameter_bindings
                .into_iter()
                .map(|(_, variable_ref)| variable_ref)
                .collect(),
            captures,
            parameter_frame,
            body: Arc::new(body),
        })
    }

    /// The variables this lambda binds, in declaration order.
    pub fn params(&self) -> &[Variable] {
        &self.params
    }

    /// The dtypes of the parameters, in declaration order.
    pub fn param_dtypes(&self) -> &[DType] {
        &self.param_dtypes
    }

    /// The lexical locations assigned to the parameters, in declaration order.
    pub fn param_refs(&self) -> &[VariableRef] {
        &self.param_refs
    }

    /// The outer lexical bindings read by this lambda body, in stable lexical order.
    pub fn captures(&self) -> &[BoundVariable] {
        &self.captures
    }

    /// The lexical frame containing the parameters.
    pub fn parameter_frame(&self) -> usize {
        self.parameter_frame
    }

    /// The bound body.
    pub fn body(&self) -> &BoundExpression {
        &self.body
    }

    /// The dtype the body evaluates to.
    pub fn body_dtype(&self) -> &DType {
        self.body.dtype()
    }

    /// The outer lexical bindings read by this lambda body.
    pub fn free_variables(&self) -> Vec<VariableRef> {
        self.captures
            .iter()
            .map(BoundVariable::variable_ref)
            .collect()
    }

    fn take_body(&mut self) -> Option<BoundExpression> {
        Arc::try_unwrap(std::mem::replace(
            &mut self.body,
            Arc::new(BoundExpression::new_root(DType::Null)),
        ))
        .ok()
    }
}

fn collect_captures(expression: &BoundExpression, parameter_frame: usize) -> Box<[BoundVariable]> {
    fn collect(
        expression: &BoundExpression,
        parameter_frame: usize,
        captures: &mut Vec<BoundVariable>,
    ) {
        match expression {
            BoundExpression::Variable(variable)
                if variable.variable_ref().frame() < parameter_frame
                    && !captures
                        .iter()
                        .any(|capture| capture.variable_ref() == variable.variable_ref()) =>
            {
                captures.push(variable.clone());
            }
            BoundExpression::Scalar { children, .. }
            | BoundExpression::ListTransform { children, .. } => {
                for child in children.iter() {
                    collect(child, parameter_frame, captures);
                }
            }
            BoundExpression::Lambda(_)
            | BoundExpression::Root { .. }
            | BoundExpression::Variable(_) => {}
        }
    }

    let mut captures = Vec::new();
    collect(expression, parameter_frame, &mut captures);
    captures.sort_by_key(|capture| {
        let reference = capture.variable_ref();
        (reference.frame(), reference.slot())
    });
    captures.into_boxed_slice()
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
            (
                BoundExpression::ListTransform {
                    dtype: lhs_dtype,
                    lambda: lhs_lambda,
                    children: lhs_children,
                },
                BoundExpression::ListTransform {
                    dtype: rhs_dtype,
                    lambda: rhs_lambda,
                    children: rhs_children,
                },
            ) => {
                lhs_dtype == rhs_dtype
                    && lhs_lambda == rhs_lambda
                    && Arc::ptr_eq(lhs_children, rhs_children)
            }
            (BoundExpression::Variable(lhs), BoundExpression::Variable(rhs)) => lhs == rhs,
            (BoundExpression::Root { .. }, _)
            | (BoundExpression::Scalar { .. }, _)
            | (BoundExpression::Lambda(_), _)
            | (BoundExpression::ListTransform { .. }, _)
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
            BoundExpression::Lambda(lambda) => {
                state.write_u8(3);
                lambda.hash(state);
            }
            BoundExpression::ListTransform {
                lambda, children, ..
            } => {
                state.write_u8(4);
                lambda.hash(state);
                Arc::as_ptr(children).hash(state);
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
            children.iter().all(|child| !child.is_lambda()),
            "a scalar function cannot take a lambda as an ordinary argument"
        );

        let arg_dtypes = children
            .iter()
            .map(|child| child.dtype().clone())
            .collect_vec();
        let dtype = scalar_fn.return_dtype(&arg_dtypes)?;

        Ok(Self::Scalar {
            dtype,
            scalar_fn,
            children: children.into(),
        })
    }

    /// Create a typed dedicated list-transform node from its already-bound outer children.
    pub(crate) fn try_new_list_transform(
        list: BoundExpression,
        lambda: BoundLambda,
        captures: impl IntoIterator<Item = BoundExpression>,
    ) -> VortexResult<Self> {
        let captures = captures.into_iter().collect::<Vec<_>>();
        vortex_ensure!(
            captures.len() == lambda.captures().len(),
            "list_transform() lambda requires {} captures, got {}",
            lambda.captures().len(),
            captures.len()
        );
        for (index, (capture, expected)) in captures.iter().zip(lambda.captures()).enumerate() {
            vortex_ensure!(
                capture.dtype() == expected.dtype(),
                "list_transform() capture {index} expects dtype {}, got {}",
                expected.dtype(),
                capture.dtype()
            );
        }
        let dtype = output_dtype(list.dtype(), lambda.body_dtype())?;
        let children = std::iter::once(list).chain(captures).collect();
        Ok(Self::ListTransform {
            dtype,
            lambda,
            children: Arc::new(children),
        })
    }

    /// Rebuild this node with new bound children, recomputing its dtype.
    pub fn with_children(
        self,
        children: impl IntoIterator<Item = BoundExpression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        match &self {
            BoundExpression::Scalar { scalar_fn, .. } => {
                Self::try_new_vec(scalar_fn.clone(), children)
            }
            BoundExpression::ListTransform { lambda, .. } => {
                let Some((list, captures)) = children.split_first() else {
                    vortex_bail!("list_transform() requires a list child");
                };
                Self::try_new_list_transform(list.clone(), lambda.clone(), captures.to_vec())
            }
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
    pub fn dtype(&self) -> &DType {
        match self {
            Self::Scalar { dtype, .. }
            | Self::ListTransform { dtype, .. }
            | Self::Root { dtype } => dtype,
            Self::Lambda(lambda) => lambda.body_dtype(),
            Self::Variable(variable) => variable.dtype(),
        }
    }

    /// The ordinary bound children of this node, in argument order.
    ///
    /// A bound lambda body is available through [`BoundLambda::body`] instead.
    pub fn children(&self) -> &[BoundExpression] {
        match self {
            Self::Scalar { children, .. } | Self::ListTransform { children, .. } => {
                children.as_slice()
            }
            Self::Lambda(_) | Self::Root { .. } | Self::Variable(_) => &[],
        }
    }

    /// Return the child at `index`.
    pub fn child(&self, index: usize) -> &BoundExpression {
        &self.children()[index]
    }

    /// The scalar function for this node, or `None` if it is a root or variable.
    pub fn as_scalar(&self) -> Option<&ScalarFnRef> {
        match self {
            Self::Scalar { scalar_fn, .. } => Some(scalar_fn),
            Self::Lambda(_)
            | Self::ListTransform { .. }
            | Self::Root { .. }
            | Self::Variable(_) => None,
        }
    }

    /// Return this node's bound lambda, if it is a lambda.
    pub fn as_lambda(&self) -> Option<&BoundLambda> {
        match self {
            Self::Lambda(lambda) => Some(lambda),
            Self::Scalar { .. }
            | Self::ListTransform { .. }
            | Self::Root { .. }
            | Self::Variable(_) => None,
        }
    }

    /// The lambda owned by a list transform node, if this is one.
    pub fn as_list_transform(&self) -> Option<(&BoundLambda, &[BoundExpression])> {
        match self {
            Self::ListTransform {
                lambda, children, ..
            } => Some((lambda, children)),
            Self::Lambda(_) | Self::Scalar { .. } | Self::Root { .. } | Self::Variable(_) => None,
        }
    }

    /// Whether this node is a bound lambda.
    pub fn is_lambda(&self) -> bool {
        self.as_lambda().is_some()
    }

    /// Return this node's bound variable, if it is a variable.
    pub fn as_variable(&self) -> Option<&BoundVariable> {
        match self {
            Self::Variable(variable) => Some(variable),
            Self::Lambda(_)
            | Self::ListTransform { .. }
            | Self::Scalar { .. }
            | Self::Root { .. } => None,
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
    /// Panics when this node is not a scalar or uses a different scalar-function vtable.
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
    pub fn is_root_bound_to(&self, dtype: &DType) -> bool {
        let mut is_bound_to = true;
        pre_order_visit_down(self, |node| {
            if node.is_root() && node.dtype() != dtype {
                is_bound_to = false;
                return Ok(TraversalOrder::Stop);
            }
            Ok(TraversalOrder::Continue)
        })
        .vortex_expect("bound expression traversal cannot not fail");
        is_bound_to
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
            Self::ListTransform {
                lambda, children, ..
            } => write!(f, "list_transform({}, {lambda})", children[0]),
            Self::Root { .. } => f.write_str("$"),
            Self::Variable(variable) => write!(f, "${variable}"),
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
    pub fn bind_scope(&self, scope: &Scope) -> VortexResult<BoundExpression> {
        match self {
            Expression::Root => Ok(BoundExpression::new_root(scope.root().clone())),
            Expression::Variable(variable) => {
                let Some((dtype, variable_ref)) = scope.resolve(variable) else {
                    vortex_bail!("variable '{variable}' has no binder");
                };
                Ok(BoundExpression::Variable(BoundVariable {
                    dtype: dtype.clone(),
                    variable: variable.clone(),
                    variable_ref,
                }))
            }
            Expression::Lambda(_) => {
                vortex_bail!("a lambda can be bound only as an argument to a higher-order function")
            }
            Expression::ListTransform { children } => {
                let list = children[0].bind_scope(scope)?;
                let lambda = children[1].as_lambda().ok_or_else(|| {
                    vortex_error::vortex_err!("list_transform() requires a lambda")
                })?;
                let element_dtype = match list.dtype() {
                    DType::List(element, _) | DType::FixedSizeList(element, ..) => {
                        element.as_ref().clone()
                    }
                    dtype => vortex_bail!(
                        "list_transform() requires List, ListView, or FixedSizeList, got {dtype}"
                    ),
                };
                vortex_ensure!(
                    matches!(lambda.params().len(), 1 | 2),
                    "list_transform() lambda must take one or two parameters, got {}",
                    lambda.params().len()
                );
                let parameter_dtypes = std::iter::once(element_dtype.clone()).chain(
                    (lambda.params().len() == 2).then_some(DType::Primitive(
                        crate::dtype::PType::U64,
                        crate::dtype::Nullability::NonNullable,
                    )),
                );
                let lambda_scope = scope
                    .clone()
                    .with_root(element_dtype)
                    .with_bindings(lambda.params().iter().cloned().zip(parameter_dtypes))?;
                let lambda = BoundLambda::bind(lambda, &lambda_scope)?;
                let captures = lambda
                    .captures()
                    .iter()
                    .cloned()
                    .map(BoundExpression::Variable)
                    .collect::<Vec<_>>();
                BoundExpression::try_new_list_transform(list, lambda, captures)
            }
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

/// Iterative drop to avoid stack overflows on deep trees.
impl Drop for BoundExpression {
    fn drop(&mut self) {
        let mut to_drop = Vec::new();
        match self {
            Self::Scalar { children, .. } => {
                if let Some(children) = Arc::get_mut(children) {
                    to_drop.append(children);
                }
            }
            Self::ListTransform {
                lambda, children, ..
            } => {
                if let Some(children) = Arc::get_mut(children) {
                    to_drop.append(children);
                }
                if let Some(body) = lambda.take_body() {
                    to_drop.push(body);
                }
            }
            Self::Lambda(lambda) => {
                if let Some(body) = lambda.take_body() {
                    to_drop.push(body);
                }
            }
            Self::Root { .. } | Self::Variable(_) => return,
        }

        while let Some(mut child) = to_drop.pop() {
            match &mut child {
                BoundExpression::Scalar { children, .. } => {
                    if let Some(grandchildren) = Arc::get_mut(children) {
                        to_drop.append(grandchildren);
                    }
                }
                BoundExpression::ListTransform {
                    lambda, children, ..
                } => {
                    if let Some(grandchildren) = Arc::get_mut(children) {
                        to_drop.append(grandchildren);
                    }
                    if let Some(body) = lambda.take_body() {
                        to_drop.push(body);
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
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use super::*;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::is_not_null;
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
        let bound = root().bind_scope(&scope())?;
        assert!(bound.is_root());
        assert_eq!(bound.dtype(), &struct_dtype());
        assert_eq!(bound, BoundExpression::new_root(struct_dtype()));
        Ok(())
    }

    #[test]
    fn every_node_carries_its_dtype() -> VortexResult<()> {
        let expr = eq(col("a"), lit(1_i32));
        let bound = expr.bind_scope(&scope())?;

        assert_eq!(bound.dtype(), &DType::Bool(Nullability::NonNullable));

        let lhs = &bound.children()[0];
        assert_eq!(
            lhs.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(lhs.children()[0].dtype(), &struct_dtype());
        Ok(())
    }

    #[test]
    fn bind_agrees_with_return_dtype() -> VortexResult<()> {
        for expr in [root(), col("a"), eq(col("a"), lit(1_i32)), lit(true)] {
            assert_eq!(
                expr.bind(&struct_dtype())?.dtype(),
                &expr.return_dtype(&struct_dtype())?,
                "disagreement for {expr}"
            );
        }
        Ok(())
    }

    #[test]
    fn contains_scalar_function() -> VortexResult<()> {
        let bound = eq(col("a"), lit(1_i32)).bind_scope(&scope())?;
        assert!(bound.contains::<Literal>()?);
        assert!(!root().bind_scope(&scope())?.contains::<Literal>()?);
        Ok(())
    }

    #[test]
    fn bound_to_checks_every_root() -> VortexResult<()> {
        let dtype = struct_dtype();
        let bound = eq(col("a"), col("a")).bind(&dtype)?;
        assert!(bound.is_root_bound_to(&dtype));
        assert!(!bound.is_root_bound_to(&DType::Bool(Nullability::NonNullable)));
        assert!(
            lit(true)
                .bind(&dtype)?
                .is_root_bound_to(&DType::Bool(Nullability::NonNullable))
        );
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
    fn variable_binds_to_its_scope() -> VortexResult<()> {
        let value_dtype = DType::Primitive(PType::I64, Nullability::Nullable);
        let scope = scope().with_bindings([(Variable::new("value"), value_dtype.clone())])?;
        let expression = var("value");

        assert!(expression.return_dtype(scope.root()).is_err());

        let bound = expression.bind_scope(&scope)?;
        let variable = bound
            .as_variable()
            .vortex_expect("variable must remain a variable after binding");
        assert_eq!(bound.dtype(), &value_dtype);
        assert_eq!(variable.variable(), &Variable::new("value"));
        assert_eq!(variable.variable_ref().frame(), 0);
        assert_eq!(variable.variable_ref().slot(), 0);
        assert_eq!(bound.to_string(), "$value");
        Ok(())
    }

    #[test]
    fn unbound_variable_is_rejected() {
        assert!(var("missing").bind_scope(&scope()).is_err());
    }

    #[test]
    fn variable_validity_is_deferred_until_binding() -> VortexResult<()> {
        let value_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let scope = scope().with_bindings([(Variable::new("value"), value_dtype)])?;

        let validity = var("value").validity()?;
        assert_eq!(validity, is_not_null(var("value")));
        assert!(validity.contains::<IsNotNull>()?);
        assert_eq!(
            validity.bind_scope(&scope)?.dtype(),
            &DType::Bool(Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn lambda_signature_comes_from_its_parameter_frame() -> VortexResult<()> {
        let expression = lambda(["value"], var("value"))?;
        let value_dtype = DType::Primitive(PType::I64, Nullability::Nullable);
        let lambda_scope =
            scope().with_bindings([(Variable::new("value"), value_dtype.clone())])?;

        assert!(expression.return_dtype(&struct_dtype()).is_err());
        assert!(expression.bind_scope(&lambda_scope).is_err());

        let lambda = expression
            .as_lambda()
            .vortex_expect("the lambda factory must produce lambda syntax");
        let bound = BoundLambda::bind(lambda, &lambda_scope)?;

        assert_eq!(bound.params(), &[Variable::new("value")]);
        assert_eq!(bound.param_dtypes(), std::slice::from_ref(&value_dtype));
        assert_eq!(bound.parameter_frame(), 0);
        assert_eq!(bound.param_refs()[0].frame(), 0);
        assert_eq!(bound.param_refs()[0].slot(), 0);
        assert_eq!(bound.body_dtype(), &value_dtype);
        assert_eq!(
            bound
                .body()
                .as_variable()
                .vortex_expect("the lambda body must resolve its parameter")
                .variable_ref(),
            bound.param_refs()[0]
        );
        Ok(())
    }

    #[test]
    fn lambda_parameter_must_be_in_the_innermost_frame() -> VortexResult<()> {
        let expression = lambda(["value"], var("value"))?;
        let lambda = expression
            .as_lambda()
            .vortex_expect("the lambda factory must produce lambda syntax");
        let scope = scope()
            .with_bindings([(Variable::new("value"), DType::Null)])?
            .with_bindings([(Variable::new("other"), DType::Null)])?;

        assert!(BoundLambda::bind(lambda, &scope).is_err());
        Ok(())
    }

    #[test]
    fn lambda_tracks_outer_captures_separately_from_parameters() -> VortexResult<()> {
        let expression = lambda(["parameter"], eq(var("captured"), var("parameter")))?;
        let lambda = expression
            .as_lambda()
            .vortex_expect("the lambda factory must produce lambda syntax");
        let value_dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let scope = scope()
            .with_bindings([(Variable::new("captured"), value_dtype.clone())])?
            .with_bindings([(Variable::new("parameter"), value_dtype)])?;

        let bound = BoundLambda::bind(lambda, &scope)?;

        assert_eq!(bound.parameter_frame(), 1);
        assert_eq!(bound.param_refs().len(), 1);
        assert_eq!(bound.param_refs()[0].frame(), 1);
        assert_eq!(bound.param_refs()[0].slot(), 0);
        let captures = bound.free_variables();
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0].frame(), 0);
        assert_eq!(captures[0].slot(), 0);
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
        assert_ne!(ExactBoundExpr(bound), ExactBoundExpr(independently_bound));
        Ok(())
    }

    #[test]
    fn binding_reports_a_type_error() {
        let expr = eq(col("a"), lit("nope"));
        assert!(expr.bind_scope(&scope()).is_err());
    }
}

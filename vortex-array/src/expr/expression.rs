// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::expr::display::DisplayTreeExpr;
use crate::expr::placeholder::PlaceholderRef;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;

/// A bound expression tree.
///
/// Bound means every node is resolved against a known evaluation scope and knows its dtype.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundExpr {
    /// The evaluation scope (`$`), bound to the scope's dtype.
    Root(DType),
    /// A typed literal value embedded in the tree.
    Literal(Scalar),
    /// A value supplied by the execution context.
    Placeholder(PlaceholderRef),
    /// A call to an already-selected scalar function.
    Call(BoundCall),
}

/// A bound scalar function call.
#[derive(Clone, Debug)]
pub struct BoundCall {
    function: ScalarFnRef,
    args: Arc<[BoundExpr]>,
    return_dtype: DType,
}

impl PartialEq for BoundCall {
    fn eq(&self, other: &Self) -> bool {
        self.function == other.function && self.args == other.args
    }
}

impl Eq for BoundCall {}

impl Hash for BoundCall {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // `return_dtype` is derived from `(function, args)` at construction time, so it is
        // intentionally excluded from structural hashing.
        self.function.hash(state);
        self.args.hash(state);
    }
}

impl BoundCall {
    /// Create a new bound scalar function call and resolve its return dtype.
    pub fn try_new(
        function: ScalarFnRef,
        args: impl IntoIterator<Item = BoundExpr>,
    ) -> VortexResult<Self> {
        let args = Vec::from_iter(args);

        vortex_ensure!(
            function.signature().arity().matches(args.len()),
            "BoundExpr arity mismatch: expected {} children but got {}",
            function.signature().arity(),
            args.len()
        );

        let arg_dtypes: Vec<_> = args.iter().map(|arg| arg.dtype().clone()).collect();
        let return_dtype = function.return_dtype(&arg_dtypes)?;

        Ok(Self {
            function,
            args: args.into(),
            return_dtype,
        })
    }

    /// Returns the scalar function for this call.
    pub fn function(&self) -> &ScalarFnRef {
        &self.function
    }

    /// Returns this call's arguments.
    pub fn args(&self) -> &[BoundExpr] {
        &self.args
    }

    /// Returns this call's arguments.
    pub fn children(&self) -> &[BoundExpr] {
        self.args()
    }

    /// Returns the n'th argument of this call.
    pub fn child(&self, n: usize) -> &BoundExpr {
        &self.args()[n]
    }

    /// Returns this call's argument count.
    pub fn children_count(&self) -> usize {
        self.args().len()
    }

    /// Returns the shared argument storage for pointer-sensitive cache keys.
    pub fn args_arc(&self) -> &Arc<[BoundExpr]> {
        &self.args
    }

    /// Returns the dtype resolved when this call was constructed.
    pub fn return_dtype(&self) -> &DType {
        &self.return_dtype
    }
}

impl Drop for BoundCall {
    fn drop(&mut self) {
        let Some(args) = Arc::get_mut(&mut self.args) else {
            return;
        };

        let mut children_to_drop = Vec::with_capacity(args.len());
        for arg in args {
            children_to_drop.push(std::mem::replace(arg, drop_tombstone()));
        }

        while let Some(mut child) = children_to_drop.pop() {
            let BoundExpr::Call(call) = &mut child else {
                continue;
            };
            let Some(args) = Arc::get_mut(&mut call.args) else {
                continue;
            };
            for arg in args {
                children_to_drop.push(std::mem::replace(arg, drop_tombstone()));
            }
        }
    }
}

fn drop_tombstone() -> BoundExpr {
    BoundExpr::Literal(Scalar::null(DType::Null))
}

impl BoundExpr {
    /// Create a call expression from a scalar function and bound children.
    pub fn try_new(
        function: ScalarFnRef,
        children: impl IntoIterator<Item = BoundExpr>,
    ) -> VortexResult<Self> {
        Ok(Self::Call(BoundCall::try_new(function, children)?))
    }

    /// Returns this expression's dtype.
    pub fn dtype(&self) -> &DType {
        match self {
            Self::Root(dtype) => dtype,
            Self::Literal(scalar) => scalar.dtype(),
            Self::Placeholder(placeholder) => placeholder.dtype(),
            Self::Call(call) => call.return_dtype(),
        }
    }

    /// Returns the children of this expression.
    pub fn children(&self) -> &[BoundExpr] {
        match self {
            Self::Call(call) => call.args(),
            Self::Root(_) | Self::Literal(_) | Self::Placeholder(_) => &[],
        }
    }

    /// Returns the n'th child of this expression.
    pub fn child(&self, n: usize) -> &BoundExpr {
        &self.children()[n]
    }

    /// Replace this expression's children with the provided new children.
    pub fn with_children(
        self,
        children: impl IntoIterator<Item = BoundExpr>,
    ) -> VortexResult<Self> {
        match self {
            Self::Call(call) => BoundCall::try_new(call.function.clone(), children).map(Self::Call),
            Self::Root(_) | Self::Literal(_) | Self::Placeholder(_) => {
                let children = Vec::from_iter(children);
                vortex_ensure!(
                    children.is_empty(),
                    "BoundExpr leaf expected 0 children but got {}",
                    children.len()
                );
                Ok(self)
            }
        }
    }

    /// Returns this expression as a scalar function call if it is one.
    pub fn as_call(&self) -> Option<&BoundCall> {
        match self {
            Self::Call(call) => Some(call),
            Self::Root(_) | Self::Literal(_) | Self::Placeholder(_) => None,
        }
    }

    /// Returns whether this expression is the root scope.
    pub fn is_root(&self) -> bool {
        matches!(self, Self::Root(_))
    }

    /// Returns this expression as a literal scalar if it is one.
    pub fn as_literal(&self) -> Option<&Scalar> {
        match self {
            Self::Literal(scalar) => Some(scalar),
            Self::Root(_) | Self::Placeholder(_) | Self::Call(_) => None,
        }
    }

    /// Returns this expression as a placeholder if it is one.
    pub fn as_placeholder(&self) -> Option<&PlaceholderRef> {
        match self {
            Self::Placeholder(placeholder) => Some(placeholder),
            Self::Root(_) | Self::Literal(_) | Self::Call(_) => None,
        }
    }

    /// Returns whether this node itself is null-sensitive.
    pub fn is_null_sensitive(&self) -> bool {
        match self {
            Self::Root(_) | Self::Literal(_) | Self::Placeholder(_) => false,
            Self::Call(call) => call.function().signature().is_null_sensitive(),
        }
    }

    /// Returns whether this node itself is semantically fallible.
    pub fn is_fallible(&self) -> bool {
        match self {
            Self::Root(_) | Self::Literal(_) | Self::Placeholder(_) => false,
            Self::Call(call) => call.function().signature().is_fallible(),
        }
    }

    /// Returns a new expression representing the validity mask output of this expression.
    pub fn validity(&self) -> VortexResult<BoundExpr> {
        match self {
            Self::Literal(scalar) => Ok(crate::expr::lit(scalar.is_valid())),
            Self::Root(_) | Self::Placeholder(_) => Ok(crate::expr::is_not_null(self.clone())),
            Self::Call(call) => Ok(call
                .function()
                .validity(call)?
                .unwrap_or_else(|| crate::expr::is_not_null(self.clone()))),
        }
    }

    /// Returns an expression that proves this predicate is definitely false from stats.
    pub fn falsify(&self, session: &VortexSession) -> VortexResult<Option<BoundExpr>> {
        crate::stats::rewrite::StatsRewriteCtx::new(session).falsify(self)
    }

    /// Returns an expression that proves this predicate is definitely true from stats.
    pub fn satisfy(&self, session: &VortexSession) -> VortexResult<Option<BoundExpr>> {
        crate::stats::rewrite::StatsRewriteCtx::new(session).satisfy(self)
    }

    /// Format the expression as a compact string.
    pub fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root(_) => write!(f, "$"),
            Self::Literal(scalar) => write!(f, "{}", scalar),
            Self::Placeholder(placeholder) => write!(f, "{}()", placeholder.display_name()),
            Self::Call(call) => call.function().fmt_sql(call, f),
        }
    }

    /// Display the expression as a formatted tree structure.
    pub fn display_tree(&self) -> impl Display {
        DisplayTreeExpr(self)
    }
}

impl Display for BoundExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_sql(f)
    }
}

impl BoundExpr {
    pub(crate) fn call(&self) -> &BoundCall {
        self.as_call().vortex_expect("BoundExpr is not a call")
    }

    pub(crate) fn scalar_fn(&self) -> &ScalarFnRef {
        self.call().function()
    }

    pub(crate) fn is<V: ScalarFnVTable>(&self) -> bool {
        self.as_call().is_some_and(|call| call.function().is::<V>())
    }

    pub(crate) fn as_opt<V: ScalarFnVTable>(&self) -> Option<&V::Options> {
        self.as_call()
            .and_then(|call| call.function().as_opt::<V>())
    }

    pub(crate) fn as_<V: ScalarFnVTable>(&self) -> &V::Options {
        self.as_opt::<V>()
            .vortex_expect("BoundExpr call options type mismatch")
    }
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::not;
    use crate::expr::root;

    /// `BoundCall`'s iterative `Drop` must keep deep trees from overflowing the stack: a naive
    /// recursive drop of this chain would blow the default test-thread stack.
    #[test]
    fn drop_deep_tree() {
        let mut expr = root(DType::Bool(Nullability::NonNullable));
        for _ in 0..100_000 {
            expr = not(expr);
        }
        drop(expr);
    }
}

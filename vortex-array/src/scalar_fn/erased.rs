// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Type-erased scalar function ([`ScalarFnRef`]).

use std::any::type_name;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::debug_with::DebugWith;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::stats::Stat;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceCtx;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::SimplifyCtx;
use crate::scalar_fn::fns::is_not_null::IsNotNull;
use crate::scalar_fn::options::ScalarFnOptions;
use crate::scalar_fn::signature::ScalarFnSignature;
use crate::scalar_fn::typed::DynScalarFn;
use crate::scalar_fn::typed::TypedScalarFnInstance;

/// A type-erased scalar function, pairing a vtable with bound options behind a trait object.
///
/// This stores a [`ScalarFnVTable`] and its options behind an `Arc<dyn DynScalarFn>`, allowing
/// heterogeneous storage inside [`Expression`] and [`crate::arrays::ScalarFnArray`].
///
/// Use [`super::TypedScalarFnInstance::new()`] to construct, and [`super::TypedScalarFnInstance::erased()`] to
/// obtain a [`ScalarFnRef`].
#[derive(Clone)]
pub struct ScalarFnRef(pub(super) Arc<dyn DynScalarFn>);

impl ScalarFnRef {
    /// Returns the ID of this scalar function.
    pub fn id(&self) -> ScalarFnId {
        self.0.id()
    }

    /// Returns whether the scalar function is of the given vtable type.
    pub fn is<V: ScalarFnVTable>(&self) -> bool {
        self.0.as_any().is::<TypedScalarFnInstance<V>>()
    }

    /// Returns the typed options for this scalar function if it matches the given vtable type.
    pub fn as_opt<V: ScalarFnVTable>(&self) -> Option<&V::Options> {
        self.0
            .as_any()
            .downcast_ref::<TypedScalarFnInstance<V>>()
            .map(|sf| sf.options())
    }

    /// Returns the typed options for this scalar function if it matches the given vtable type.
    ///
    /// # Panics
    ///
    /// Panics if the vtable type does not match.
    pub fn as_<V: ScalarFnVTable>(&self) -> &V::Options {
        self.as_opt::<V>()
            .vortex_expect("Expression options type mismatch")
    }

    /// Downcast to the concrete [`TypedScalarFnInstance`].
    ///
    /// Returns `Err(self)` if the downcast fails.
    pub fn try_downcast<V: ScalarFnVTable>(
        self,
    ) -> Result<Arc<TypedScalarFnInstance<V>>, ScalarFnRef> {
        if self.0.as_any().is::<TypedScalarFnInstance<V>>() {
            let ptr = Arc::into_raw(self.0) as *const TypedScalarFnInstance<V>;
            Ok(unsafe { Arc::from_raw(ptr) })
        } else {
            Err(self)
        }
    }

    /// Downcast to the concrete [`TypedScalarFnInstance`].
    ///
    /// # Panics
    ///
    /// Panics if the downcast fails.
    pub fn downcast<V: ScalarFnVTable>(self) -> Arc<TypedScalarFnInstance<V>> {
        self.try_downcast::<V>()
            .map_err(|this| {
                vortex_err!(
                    "Failed to downcast ScalarFnRef {} to {}",
                    this.0.id(),
                    type_name::<V>(),
                )
            })
            .vortex_expect("Failed to downcast ScalarFnRef")
    }

    /// Try to downcast into a typed [`TypedScalarFnInstance`].
    pub fn downcast_ref<V: ScalarFnVTable>(&self) -> Option<&TypedScalarFnInstance<V>> {
        self.0.as_any().downcast_ref::<TypedScalarFnInstance<V>>()
    }

    /// The type-erased options for this scalar function.
    pub fn options(&self) -> ScalarFnOptions<'_> {
        ScalarFnOptions { inner: &*self.0 }
    }

    /// Signature information for this scalar function.
    pub fn signature(&self) -> ScalarFnSignature<'_> {
        ScalarFnSignature { inner: &*self.0 }
    }

    /// Compute the return [`DType`] of this expression given the input argument types.
    pub fn return_dtype(&self, arg_types: &[DType]) -> VortexResult<DType> {
        self.0.return_dtype(arg_types)
    }

    /// Transforms the expression into one representing the validity of this expression.
    pub fn validity(&self, expr: &Expression) -> VortexResult<Expression> {
        Ok(self.0.validity(expr)?.unwrap_or_else(|| {
            // TODO(ngates): make validity a mandatory method on VTable to avoid this fallback.
            IsNotNull.new_expr(EmptyOptions, [expr.clone()])
        }))
    }

    /// Execute the expression given the input arguments.
    pub fn execute(
        &self,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.0.execute(args, ctx)
    }

    /// Perform abstract reduction on this scalar function node.
    pub fn reduce(
        &self,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        self.0.reduce(node, ctx)
    }

    // ------------------------------------------------------------------
    // Expression-taking methods — used by expr/ module via pub(crate)
    // ------------------------------------------------------------------

    /// Format this expression in SQL-style format.
    pub(crate) fn fmt_sql(&self, expr: &Expression, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt_sql(expr, f)
    }

    /// Simplify the expression using type information.
    pub(crate) fn simplify(
        &self,
        expr: &Expression,
        ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        self.0.simplify(expr, ctx)
    }

    /// Simplify the expression without type information.
    pub(crate) fn simplify_untyped(&self, expr: &Expression) -> VortexResult<Option<Expression>> {
        self.0.simplify_untyped(expr)
    }

    /// Compute stat falsification expression.
    pub(crate) fn stat_falsification(
        &self,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        self.0.stat_falsification(expr, catalog)
    }

    /// Compute stat expression.
    pub(crate) fn stat_expression(
        &self,
        expr: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        self.0.stat_expression(expr, stat, catalog)
    }
}

impl Debug for ScalarFnRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarFnRef")
            .field("vtable", &self.0.id())
            .field("options", &DebugWith(|fmt| self.0.options_debug(fmt)))
            .finish()
    }
}

impl Display for ScalarFnRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", self.0.id())?;
        self.0.options_display(f)?;
        write!(f, ")")
    }
}

impl PartialEq for ScalarFnRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id() && self.0.options_eq(other.0.options_any())
    }
}
impl Eq for ScalarFnRef {}

impl Hash for ScalarFnRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.id().hash(state);
        self.0.options_hash(state);
    }
}

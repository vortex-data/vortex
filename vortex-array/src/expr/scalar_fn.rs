// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;

use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::debug_with::DebugWith;
use vortex_vector::Datum;

use crate::ArrayRef;
use crate::expr::EmptyOptions;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::ExprVTable;
use crate::expr::Expression;
use crate::expr::IsNull;
use crate::expr::Not;
use crate::expr::ReduceCtx;
use crate::expr::ReduceNode;
use crate::expr::ReduceNodeRef;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::options::ExpressionOptions;
use crate::expr::signature::ExpressionSignature;

/// An instance of an expression bound to some invocation options.
pub struct ScalarFn {
    vtable: ExprVTable,
    options: Box<dyn Any + Send + Sync>,
}

impl ScalarFn {
    /// Create a new bound expression from raw vtable and options.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the provided options are compatible with the provided vtable.
    pub(super) unsafe fn new_unchecked(
        vtable: ExprVTable,
        options: Box<dyn Any + Send + Sync>,
    ) -> Self {
        Self { vtable, options }
    }

    /// Create a new bound expression from a vtable.
    pub fn new<V: VTable>(vtable: V, options: V::Options) -> Self {
        let vtable = ExprVTable::new::<V>(vtable);
        let options = Box::new(options);
        Self { vtable, options }
    }

    /// Create a new expression from a static vtable.
    pub fn new_static<V: VTable>(vtable: &'static V, options: V::Options) -> Self {
        let vtable = ExprVTable::new_static::<V>(vtable);
        let options = Box::new(options);
        Self { vtable, options }
    }

    /// The vtable for this expression.
    pub fn vtable(&self) -> &ExprVTable {
        &self.vtable
    }

    /// Returns the ID of this expression.
    pub fn id(&self) -> ExprId {
        self.vtable.id()
    }

    /// The type-erased options for this expression.
    pub fn options(&self) -> ExpressionOptions<'_> {
        ExpressionOptions {
            vtable: &self.vtable,
            options: self.options.deref(),
        }
    }

    /// Returns whether the scalar function is of the given vtable type.
    pub fn is<V: VTable>(&self) -> bool {
        self.vtable.is::<V>()
    }

    /// Returns the typed options for this `ScalarFn` if it matches the given vtable type.
    pub fn as_opt<V: VTable>(&self) -> Option<&V::Options> {
        self.vtable.is::<V>().then(|| {
            self.options()
                .as_any()
                .downcast_ref::<V::Options>()
                .vortex_expect("Expression options type mismatch")
        })
    }

    /// Returns the typed options for this `ScalarFn` if it matches the given vtable type.
    pub fn as_<V: VTable>(&self) -> &V::Options {
        self.as_opt::<V>()
            .vortex_expect("Expression options type mismatch")
    }
    /// Signature information for this expression.
    pub fn signature(&self) -> ExpressionSignature<'_> {
        ExpressionSignature {
            vtable: &self.vtable,
            options: self.options.deref(),
        }
    }

    /// Compute the return [`DType`] of this expression given the input argument types.
    pub fn return_dtype(&self, arg_types: &[DType]) -> VortexResult<DType> {
        self.vtable
            .as_dyn()
            .return_dtype(self.options.deref(), arg_types)
    }

    /// Evaluate the expression, returning an ArrayRef.
    ///
    /// NOTE: this function will soon be deprecated as all expressions will evaluate trivially
    ///  into an ExprArray.
    pub fn evaluate(&self, expr: &Expression, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        self.vtable.as_dyn().evaluate(expr, scope)
    }

    /// Transforms the expression into one representing the validity of this expression.
    pub fn validity(&self, expr: &Expression) -> VortexResult<Expression> {
        Ok(self.vtable.as_dyn().validity(expr)?.unwrap_or_else(|| {
            // TODO(ngates): make validity a mandatory method on VTable to avoid this fallback.
            // TODO(ngates): add an IsNotNull expression.
            Not.new_expr(
                EmptyOptions,
                [IsNull.new_expr(EmptyOptions, [expr.clone()])],
            )
        }))
    }

    /// Execute the expression given the input arguments.
    pub fn execute(&self, ctx: ExecutionArgs) -> VortexResult<Datum> {
        self.vtable.as_dyn().execute(self.options.deref(), ctx)
    }

    /// Perform abstract reduction on this scalar function node.
    pub fn reduce(
        &self,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        self.vtable.as_dyn().reduce(self.options.deref(), node, ctx)
    }
}

impl Clone for ScalarFn {
    fn clone(&self) -> Self {
        ScalarFn {
            vtable: self.vtable.clone(),
            options: self.vtable.as_dyn().options_clone(self.options.deref()),
        }
    }
}

impl Debug for ScalarFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundExpression")
            .field("vtable", &self.vtable)
            .field(
                "options",
                &DebugWith(|fmt| {
                    self.vtable
                        .as_dyn()
                        .options_debug(self.options.deref(), fmt)
                }),
            )
            .finish()
    }
}

impl Display for ScalarFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", self.vtable.id())?;
        self.vtable
            .as_dyn()
            .options_display(self.options.deref(), f)?;
        write!(f, ")")
    }
}

impl PartialEq for ScalarFn {
    fn eq(&self, other: &Self) -> bool {
        self.vtable == other.vtable
            && self
                .vtable
                .as_dyn()
                .options_eq(self.options.deref(), other.options.deref())
    }
}
impl Eq for ScalarFn {}

impl Hash for ScalarFn {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vtable.hash(state);
        self.vtable
            .as_dyn()
            .options_hash(self.options.deref(), state);
    }
}

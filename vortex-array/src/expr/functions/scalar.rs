// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_utils::debug_with::DebugWith;
use vortex_vector::Datum;

use crate::expr::functions::execution::ExecutionCtx;
use crate::expr::functions::Arity;
use crate::expr::functions::FunctionId;
use crate::expr::functions::NullHandling;
use crate::expr::functions::ScalarFnVTable;
use crate::expr::functions::{ArgName, VTable};
use crate::expr::stats::Stat;
use crate::expr::Expression;
use crate::expr::StatsCatalog;

/// An instance of a scalar function bound to some invocation options.
pub struct ScalarFn {
    vtable: ScalarFnVTable,
    options: Box<dyn Any + Send + Sync>,
}

impl ScalarFn {
    /// Create a new scalar function instance.
    pub fn new<V: VTable>(vtable: V, options: V::Options) -> ScalarFn {
        let vtable = ScalarFnVTable::new::<V>(vtable);
        let options = Box::new(options);
        ScalarFn { vtable, options }
    }

    /// Create a new scalar function instance from a static vtable.
    pub fn new_static<V: VTable>(vtable: &'static V, options: V::Options) -> ScalarFn {
        let vtable = ScalarFnVTable::new_static(vtable);
        let options = Box::new(options);
        ScalarFn { vtable, options }
    }

    /// Create a new scalar function instance.
    ///
    /// # Safety
    ///
    /// The options must be of the correct type for the given vtable.
    pub(crate) unsafe fn new_unchecked(
        vtable: ScalarFnVTable,
        options: Box<dyn Any + Send + Sync>,
    ) -> Self {
        Self { vtable, options }
    }

    /// Return the function ID for this scalar function.
    pub fn id(&self) -> FunctionId {
        self.vtable.id()
    }

    /// Return the vtable of this scalar function.
    pub fn vtable(&self) -> &ScalarFnVTable {
        &self.vtable
    }

    /// Get the options for this scalar function.
    pub fn options(&self) -> ScalarFnOptions<'_> {
        ScalarFnOptions {
            vtable: &self.vtable,
            options: self.options.as_ref(),
        }
    }

    /// Return the signature information for this scalar function.
    pub fn signature(&self) -> ScalarFnSignature<'_> {
        ScalarFnSignature {
            vtable: &self.vtable,
            options: self.options.as_ref(),
        }
    }

    pub fn stat_falsification(
        &self,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        self.vtable
            .as_dyn()
            .stat_falsification(self.options.as_ref(), expr, catalog)
    }

    pub fn stat_expression(
        &self,
        expr: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        self.vtable
            .as_dyn()
            .stat_expression(self.options.as_ref(), expr, stat, catalog)
    }

    pub fn return_dtype(&self, arg_types: &[DType]) -> VortexResult<DType> {
        self.vtable
            .as_dyn()
            .return_dtype(self.options.as_ref(), arg_types)
    }

    pub fn execute(&self, ctx: &ExecutionCtx) -> VortexResult<Datum> {
        self.vtable.as_dyn().execute(self.options.as_ref(), ctx)
    }
}

impl Clone for ScalarFn {
    fn clone(&self) -> Self {
        Self {
            vtable: self.vtable.clone(),
            options: self.vtable.as_dyn().options_clone(self.options.as_ref()),
        }
    }
}

impl Debug for ScalarFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarFn")
            .field("id", &self.vtable.id())
            .field(
                "options",
                &DebugWith(|fmt| {
                    self.vtable
                        .as_dyn()
                        .options_debug(self.options.as_ref(), fmt)
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
            .options_display(self.options.as_ref(), f)?;
        write!(f, ")")
    }
}

impl PartialEq for ScalarFn {
    fn eq(&self, other: &Self) -> bool {
        self.vtable.id() == other.vtable.id()
            && self
                .vtable
                .as_dyn()
                .options_eq(self.options.as_ref(), other.options.as_ref())
    }
}
impl Eq for ScalarFn {}

impl Hash for ScalarFn {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vtable.id().hash(state);
        self.vtable
            .as_dyn()
            .options_hash(self.options.as_ref(), state);
    }
}

/// Opaque reference to scalar function signature information.
pub struct ScalarFnSignature<'a> {
    pub(crate) vtable: &'a ScalarFnVTable,
    pub(crate) options: &'a dyn Any,
}

impl ScalarFnSignature<'_> {
    pub fn arity(&self) -> Arity {
        self.vtable.as_dyn().arity(self.options)
    }

    pub fn null_handling(&self) -> NullHandling {
        self.vtable.as_dyn().null_handling(self.options)
    }

    pub fn arg_name(&self, arg_idx: usize) -> ArgName {
        self.vtable.as_dyn().arg_name(self.options, arg_idx)
    }
}

/// Opaque reference to scalar function options.
pub struct ScalarFnOptions<'a> {
    pub(crate) vtable: &'a ScalarFnVTable,
    pub(crate) options: &'a dyn Any,
}

impl ScalarFnOptions<'_> {
    /// Get the options as a `dyn Any`.
    pub fn as_any(&self) -> &dyn Any {
        self.options
    }
}

impl Display for ScalarFnOptions<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.vtable.as_dyn().options_display(self.options, f)
    }
}

impl Debug for ScalarFnOptions<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.vtable.as_dyn().options_debug(self.options, f)
    }
}

impl ScalarFnOptions<'_> {
    /// Serializes the options into a byte vector.
    pub fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        self.vtable.as_dyn().options_serialize(self.options)
    }
}

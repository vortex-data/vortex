// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::hash::Hasher;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_utils::debug_with::DebugWith;
use vortex_vector::Datum;

use crate::functions::execution::ExecutionCtx;
use crate::functions::ScalarFnVTable;

/// An instance of a scalar function bound to some invocation options.
pub struct ScalarFn {
    vtable: ScalarFnVTable,
    options: Box<dyn Any + Send + Sync>,
}

impl Clone for ScalarFn {
    fn clone(&self) -> Self {
        Self {
            vtable: self.vtable.clone(),
            options: self.vtable.as_dyn().clone_options(self.options.as_ref()),
        }
    }
}

impl Debug for ScalarFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarFn")
            .field("id", &self.vtable.id())
            .field(
                "options",
                &DebugWith(|fmt| {
                    self.vtable
                        .as_dyn()
                        .debug_options(self.options.as_ref(), fmt)
                }),
            )
            .finish()
    }
}

impl Display for ScalarFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", self.vtable.id())?;
        self.vtable.as_dyn().fmt_options(self.options.as_ref(), f)?;
        write!(f, ")")
    }
}

impl PartialEq for ScalarFn {
    fn eq(&self, other: &Self) -> bool {
        self.vtable.id() == other.vtable.id()
            && self
                .vtable
                .as_dyn()
                .eq_options(self.options.as_ref(), other.options.as_ref())
    }
}

impl Eq for ScalarFn {}

impl Hash for ScalarFn {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vtable.id().hash(state);
        self.vtable
            .as_dyn()
            .hash_options(self.options.as_ref(), state);
    }
}

impl ScalarFn {
    pub(crate) unsafe fn new_unchecked(
        vtable: ScalarFnVTable,
        options: Box<dyn Any + Send + Sync>,
    ) -> Self {
        Self { vtable, options }
    }

    pub fn serialize_options(&self) -> VortexResult<Option<Vec<u8>>> {
        self.vtable
            .as_dyn()
            .serialize_options(self.options.as_ref())
    }

    pub fn fmt_options(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.vtable.as_dyn().fmt_options(self.options.as_ref(), fmt)
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

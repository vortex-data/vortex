// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::functions::vtable::DynScalarFnVTable;
use crate::functions::ScalarFnVTable;
use std::any::Any;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_utils::debug_with::DebugWith;

/// An instance of a scalar function bound to some invocation options.
pub struct ScalarFn {
    vtable: ScalarFnVTable,
    options: Box<dyn Any + Send + Sync>,
}

impl Clone for ScalarFn {
    fn clone(&self) -> Self {
        Self {
            vtable: self.vtable.clone(),
            options: self.vtable.clone_options(self.options.as_ref()),
        }
    }
}

impl Debug for ScalarFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarFn")
            .field("id", &self.vtable.id())
            .field(
                "options",
                &DebugWith(|fmt| self.vtable.debug_options(self.options.as_ref(), fmt)),
            )
            .finish()
    }
}

impl PartialEq for ScalarFn {
    fn eq(&self, other: &Self) -> bool {
        self.vtable.id() == other.vtable.id()
            && self
                .vtable
                .eq_options(self.options.as_ref(), other.options.as_ref())
    }
}

impl Eq for ScalarFn {}

impl Hash for ScalarFn {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vtable.id().hash(state);
        self.vtable.hash_options(self.options.as_ref(), state);
    }
}

impl ScalarFn {
    pub fn serialize_options(&self) -> VortexResult<Option<Vec<u8>>> {
        self.vtable.serialize_options(self.options.as_ref())
    }

    pub fn fmt_options(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.vtable.fmt_options(self.options.as_ref(), fmt)
    }

    pub fn return_dtype(&self, arg_types: &[DType]) -> VortexResult<DType> {
        self.vtable.return_dtype(self.options.as_ref(), arg_types)
    }
}

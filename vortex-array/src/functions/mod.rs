// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod execution;
pub mod funcs;
mod session;
mod signature;
mod vtable;

pub use session::*;
pub use signature::*;
pub use vtable::*;

use arcref::ArcRef;
use std::any::Any;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use vortex_utils::debug_with::DebugWith;

pub type FunctionId = ArcRef<str>;
pub type ChildName = ArcRef<str>;

/// An instance of a scalar function bound to some specific invocation options.
pub struct ScalarFunction {
    vtable: ScalarFunctionVTable,
    options: Box<dyn Any + Send + Sync>,
}

impl Clone for ScalarFunction {
    fn clone(&self) -> Self {
        Self {
            vtable: self.vtable.clone(),
            options: self.vtable.clone_options(self.options.as_ref()),
        }
    }
}

impl Debug for ScalarFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarFunction")
            .field("id", &self.vtable.id())
            .field(
                "options",
                &DebugWith(|fmt| self.vtable.debug_options(self.options.as_ref(), fmt)),
            )
            .finish()
    }
}

impl PartialEq for ScalarFunction {
    fn eq(&self, other: &Self) -> bool {
        self.vtable.id() == other.vtable.id()
            && self
                .vtable
                .eq_options(self.options.as_ref(), other.options.as_ref())
    }
}

impl Eq for ScalarFunction {}

impl Hash for ScalarFunction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vtable.id().hash(state);
        self.vtable.hash_options(self.options.as_ref(), state);
    }
}

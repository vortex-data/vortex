// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ScalarFnPlugin;

/// Information about the signature of an expression.
pub struct ScalarFnSignature<'a> {
    pub(crate) vtable: &'a ScalarFnPlugin,
    pub(crate) options: &'a dyn Any,
}

impl ScalarFnSignature<'_> {
    /// Returns the arity of this expression.
    pub fn arity(&self) -> Arity {
        self.vtable.as_dyn().arity(self.options)
    }

    /// Returns the name of the nth child of this expression.
    pub fn child_name(&self, index: usize) -> ChildName {
        self.vtable.as_dyn().child_name(self.options, index)
    }

    /// Returns whether this expression itself is null-sensitive.
    /// See [`crate::scalar_fn::ScalarFnVTable::is_null_sensitive`].
    pub fn is_null_sensitive(&self) -> bool {
        self.vtable.as_dyn().is_null_sensitive(self.options)
    }

    /// Returns whether this expression itself is fallible.
    /// See [`crate::scalar_fn::ScalarFnVTable::is_fallible`].
    pub fn is_fallible(&self) -> bool {
        self.vtable.as_dyn().is_fallible(self.options)
    }
}

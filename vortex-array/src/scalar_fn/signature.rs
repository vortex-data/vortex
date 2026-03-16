// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::typed::DynScalarFn;

/// Information about the signature of an expression.
pub struct ScalarFnSignature<'a> {
    pub(super) inner: &'a dyn DynScalarFn,
}

impl ScalarFnSignature<'_> {
    /// Returns the arity of this expression.
    pub fn arity(&self) -> Arity {
        self.inner.arity()
    }

    /// Returns the name of the nth child of this expression.
    pub fn child_name(&self, index: usize) -> ChildName {
        self.inner.child_name(index)
    }

    /// Returns whether this expression itself is null-sensitive.
    /// See [`crate::scalar_fn::ScalarFnVTable::is_null_sensitive`].
    pub fn is_null_sensitive(&self) -> bool {
        self.inner.is_null_sensitive()
    }

    /// Returns whether this expression itself is fallible.
    /// See [`crate::scalar_fn::ScalarFnVTable::is_fallible`].
    pub fn is_fallible(&self) -> bool {
        self.inner.is_fallible()
    }
}

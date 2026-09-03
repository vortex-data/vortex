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

    /// Returns whether this expression itself is strict.
    /// See [`crate::scalar_fn::ScalarFnVTable::is_strict`].
    pub fn is_strict(&self) -> bool {
        self.inner.is_strict()
    }

    /// Returns whether this expression itself is infallible.
    /// See [`crate::scalar_fn::ScalarFnVTable::is_infallible`].
    pub fn is_infallible(&self) -> bool {
        self.inner.is_infallible()
    }
}

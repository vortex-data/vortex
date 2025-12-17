// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExprVTable;

/// Information about the signature of an expression.
pub struct ExpressionSignature<'a> {
    pub(super) vtable: &'a ExprVTable,
    pub(super) options: &'a dyn Any,
}

impl ExpressionSignature<'_> {
    /// Returns the arity of this expression.
    pub fn arity(&self) -> Arity {
        self.vtable.as_dyn().arity(self.options)
    }

    /// Returns the name of the nth child of this expression.
    pub fn child_name(&self, index: usize) -> ChildName {
        self.vtable.as_dyn().child_name(self.options, index)
    }

    /// Returns whether this expression itself is null-sensitive.
    /// See [`crate::expr::VTable::is_null_sensitive`].
    pub fn is_null_sensitive(&self) -> bool {
        self.vtable.as_dyn().is_null_sensitive(self.options)
    }

    /// Returns whether this expression itself is fallible.
    /// See [`crate::expr::VTable::is_fallible`].
    pub fn is_fallible(&self) -> bool {
        self.vtable.as_dyn().is_fallible(self.options)
    }

    /// Return if the expression add or remove a structural wrapper e.g. struct or list.
    pub fn is_structural(&self) -> bool {
        self.vtable.as_dyn().is_structural(self.options)
    }
}

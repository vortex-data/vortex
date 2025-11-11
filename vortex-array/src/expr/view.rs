// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use vortex_error::VortexExpect;

use crate::expr::{Expression, VTable};

/// A view over an [`Expression`] with an associated vtable, allowing typed access to the
/// expression's instance data.
pub struct ExpressionView<'a, V: VTable> {
    expression: &'a Expression,
    vtable: &'a V,
    data: &'a V::Instance,
}

impl<'a, V: VTable> ExpressionView<'a, V> {
    /// Wrap up the given expression as an [`ExpressionView`] of the specified vtable type.
    ///
    /// # Panics
    ///
    /// Panics if the expression cannot be downcast to the specified vtable type.
    #[inline]
    pub fn new(expression: &'a Expression) -> Self {
        Self::maybe_new(expression).vortex_expect("Failed to downcast expression")
    }

    /// Attempts to wrap up the given expression as an [`ExpressionView`] of the specified vtable type.
    #[inline]
    pub fn maybe_new(expression: &'a Expression) -> Option<Self> {
        let vtable = expression.vtable().as_opt::<V>()?;
        let data = expression.data().downcast_ref::<V::Instance>()?;
        Some(Self {
            expression,
            vtable,
            data,
        })
    }
}

impl<'a, V: VTable> ExpressionView<'a, V> {
    /// Returns the vtable for this expression.
    #[inline(always)]
    pub fn vtable(&self) -> &'a V {
        self.vtable
    }

    /// Returns the instance data for this expression.
    #[inline(always)]
    pub fn data(&self) -> &'a V::Instance {
        self.data
    }
}

impl<'a, V: VTable> Deref for ExpressionView<'a, V> {
    type Target = Expression;

    fn deref(&self) -> &Self::Target {
        self.expression
    }
}

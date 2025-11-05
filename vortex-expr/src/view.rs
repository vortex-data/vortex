// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::ops::Deref;

use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::{Expression, VTable};

/// A view over an [`Expression`] with an associated vtable, allowing typed access to the
/// expression's instance data.
pub struct ExpressionView<'a, V: VTable> {
    expression: &'a Expression,
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
        Self::try_new(expression).vortex_expect("Failed to downcast expression")
    }

    /// Attempts to wrap up the given expression as an [`ExpressionView`] of the specified vtable type.
    #[inline]
    pub fn try_new(expression: &'a Expression) -> VortexResult<Self> {
        expression.vtable().as_opt::<V>().ok_or_else(|| {
            vortex_err!(
                "Failed to downcast {} to {}",
                expression.id(),
                type_name::<V>()
            )
        })?;

        let data = expression
            .data()
            .downcast_ref::<V::Instance>()
            .ok_or_else(|| {
                vortex_err!(
                    "Failed to downcast expression instance data to expected type {}",
                    type_name::<V::Instance>()
                )
            })?;

        Ok(Self { expression, data })
    }
}

impl<'a, V: VTable> ExpressionView<'a, V> {
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

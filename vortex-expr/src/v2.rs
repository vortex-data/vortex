// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{ExprVTable, ExpressionView, ScopeVar, VTable};
use std::any::Any;
use std::sync::Arc;
use vortex_array::ArrayRef;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

/// A node in a Vortex expression tree.
///
/// Expressions represent scalar computations that can be performed on data. Each
/// expression consists of an encoding (vtable), heap-allocated metadata, and child expressions.
#[derive(Clone, Debug)]
pub struct Expression {
    /// The vtable for this expression.
    vtable: ExprVTable,
    /// The instance data for this expression.
    instance: Arc<dyn Any>,
    /// Any children of this expression.
    children: Arc<[Expression]>,
}

impl Expression {
    /// Creates a new expression with the given encoding, metadata, and children.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided `encoding` is not compatible with the
    /// `metadata` and `children` or the encoding's own validation logic fails.
    pub fn try_new(
        vtable: ExprVTable,
        instance: Arc<dyn Any>,
        children: Arc<[Expression]>,
    ) -> VortexResult<Self> {
        // Validate that the encoding is compatible with the metadata and children.
        vtable
            .as_dyn()
            .validate(instance.as_ref(), children.as_ref())?;
        Ok(Self {
            vtable,
            instance,
            children,
        })
    }

    /// Creates a new expression with the given encoding, metadata, and children.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the provided `encoding` is compatible with the
    /// `metadata` and `children`. Failure to do so may lead to undefined behavior
    ///  when the expression is used.
    pub unsafe fn new_unchecked(
        vtable: ExprVTable,
        instance: Arc<dyn Any>,
        children: Arc<[Expression]>,
    ) -> Self {
        Self {
            vtable,
            instance,
            children,
        }
    }

    /// Returns a typed view of this expression for the given vtable.
    ///
    /// # Panics
    ///
    /// Panics if the expression's encoding or metadata cannot be cast to the specified vtable.
    pub fn as_view<V: VTable>(&self) -> ExpressionView<'_, V> {
        ExpressionView::new(
            self.vtable
                .as_dyn()
                .as_any()
                .downcast_ref::<V>()
                .vortex_expect("Failed to downcast expression vtable to expected type"),
            self.instance
                .as_any()
                .downcast_ref::<V::Instance>()
                .vortex_expect("Failed to downcast expression instance to expected type"),
            &self.children,
        )
    }

    /// Returns the children of this expression.
    pub fn children(&self) -> &Arc<[Expression]> {
        &self.children
    }

    /// Replace the children of this expression with the provided new children.
    pub fn with_children(mut self, children: Arc<[Expression]>) -> VortexResult<Self> {
        self.vtable
            .as_dyn()
            .validate(self.instance.as_ref(), &children)?;
        self.children = children;
        Ok(self)
    }

    /// Computes the return dtype of this expression given the input dtype.
    pub fn return_dtype(&self, scope: &DType) -> VortexResult<DType> {
        self.vtable
            .as_dyn()
            .return_dtype(self.instance.as_ref(), self.children.as_ref(), scope)
    }

    /// Evaluates the expression in the given scope.
    pub fn evaluate(&self, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        self.vtable
            .as_dyn()
            .evaluate(self.instance.as_ref(), self.children.as_ref(), scope)
    }
}

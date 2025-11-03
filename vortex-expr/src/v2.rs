// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::metadata::ExprMetadata;
use crate::{ExprEncodingRef, VTable};
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
    encoding: ExprEncodingRef,
    metadata: Arc<dyn ExprMetadata>,
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
        encoding: ExprEncodingRef,
        metadata: Arc<dyn ExprMetadata>,
        children: Arc<[Expression]>,
    ) -> VortexResult<Self> {
        // Validate that the encoding is compatible with the metadata and children.
        encoding.validate(metadata.as_ref(), children.as_ref())?;
        Ok(Self {
            encoding,
            metadata,
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
        encoding: ExprEncodingRef,
        metadata: Arc<dyn ExprMetadata>,
        children: Arc<[Expression]>,
    ) -> Self {
        Self {
            encoding,
            metadata,
            children,
        }
    }

    /// Returns a typed view of this expression for the given vtable.
    ///
    /// # Panics
    ///
    /// Panics if the expression's encoding or metadata cannot be cast to the specified vtable.
    pub fn as_view<V: VTable>(&self) -> ExpressionView<'_, V> {
        ExpressionView {
            encoding: self.encoding.as_::<V>(),
            metadata: self
                .metadata
                .as_any()
                .downcast_ref::<V::Metadata2>()
                .vortex_expect("Failed to downcast expression metadata to expected type"),
            children: &self.children,
        }
    }

    /// Returns the children of this expression.
    pub fn children(&self) -> &Arc<[Expression]> {
        &self.children
    }

    /// Replace the children of this expression with the provided new children.
    pub fn with_children(mut self, children: Arc<[Expression]>) -> VortexResult<Self> {
        self.encoding.validate(self.metadata.as_ref(), &children)?;
        self.children = children;
        Ok(self)
    }

    /// Computes the return dtype of this expression given the input dtype.
    pub fn return_dtype(&self, scope: &DType) -> VortexResult<DType> {
        self.encoding
            .return_dtype(self.metadata.as_ref(), self.children.as_ref(), scope)
    }

    /// Evaluates the expression in the given scope.
    pub fn evaluate(&self, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        self.encoding
            .evaluate(self.metadata.as_ref(), self.children.as_ref(), scope)
    }
}

pub struct ExpressionView<'a, V: VTable> {
    encoding: &'a V::Encoding,
    metadata: &'a V::Metadata2,
    children: &'a [Expression],
}

impl<'a, V: VTable> ExpressionView<'a, V> {
    pub fn encoding(&self) -> &'a V::Encoding {
        self.encoding
    }

    pub fn metadata(&self) -> &'a V::Metadata2 {
        self.metadata
    }

    pub fn children(&self) -> &'a [Expression] {
        self.children
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::metadata::ExprMetadata;
use crate::{ExprEncodingRef, VTable};
use std::sync::Arc;

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
    pub fn try_new(
        encoding: ExprEncodingRef,
        metadata: Arc<dyn ExprMetadata>,
        children: Arc<[Expression]>,
    ) -> Result<Self, String> {
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

    /// Returns the children of this expression.
    pub fn children(&self) -> &Arc<[Expression]> {
        &self.children
    }
}

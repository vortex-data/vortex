// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod canonical;
mod operations;
mod validity;

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::buffer::BufferHandle;
use crate::expr::Expression;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::VTable;

vtable!(Expression);

#[derive(Clone, Debug)]
pub struct ExpressionArray {
    expression: Expression,
    dtype: DType,
    input: ArrayRef,
    stats: ArrayStats,
}

impl ExpressionArray {
    pub fn try_new(expression: Expression, input: ArrayRef) -> VortexResult<Self> {
        let dtype = expression.return_dtype(input.dtype())?;
        Ok(Self {
            expression,
            dtype,
            input,
            stats: ArrayStats::default(),
        })
    }

    /// Create a new expression array without performing any validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the return DType is valid for the given input.
    pub unsafe fn new_unchecked(expression: Expression, dtype: DType, input: ArrayRef) -> Self {
        Self {
            expression,
            dtype,
            input,
            stats: ArrayStats::default(),
        }
    }
}

/// VTable for the expression array.
#[derive(Debug)]
pub struct ExpressionVTable;

impl VTable for ExpressionVTable {
    type Array = ExpressionArray;
    type Metadata = Self;
    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = Self;
    type EncodeVTable = Self;

    fn id(&self) -> ArrayId {
        ArrayId::from("vortex.expression")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        ExpressionVTable.as_vtable()
    }

    fn metadata(_array: &Self::Array) -> VortexResult<Self::Metadata> {
        todo!()
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        todo!()
    }

    fn deserialize(bytes: &[u8]) -> VortexResult<Self::Metadata> {
        todo!()
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array> {
        todo!()
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        todo!()
    }
}

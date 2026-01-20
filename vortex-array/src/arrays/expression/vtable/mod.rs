// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod operations;
mod rules;
mod validity;
mod visitor;

use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Range;

use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::expression::vtable::rules::RULES;
use crate::buffer::BufferHandle;
use crate::expr::Expression;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::NotSupported;
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

    /// Get the expression associated with this array.
    pub fn expression(&self) -> &Expression {
        &self.expression
    }
}

pub struct ExpressionArrayMetadata {
    expression: Expression,
    scope_dtype: DType,
}

impl Debug for ExpressionArrayMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("")
            .field("expression", &format!("{}", self.expression))
            .field("scope_dtype", &format!("{}", self.scope_dtype))
            .finish()
    }
}

/// VTable for the expression array.
#[derive(Debug)]
pub struct ExpressionVTable;

impl ExpressionVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.expression");
}

impl VTable for ExpressionVTable {
    type Array = ExpressionArray;
    type Metadata = ExpressionArrayMetadata;
    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata> {
        Ok(ExpressionArrayMetadata {
            expression: array.expression.clone(),
            scope_dtype: array.input.dtype().clone(),
        })
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(_bytes: &[u8]) -> VortexResult<Self::Metadata> {
        // TODO(ngates): do we pass the VortexSession into the deserialize function?
        //  Or do we force the ExpressionVTable to hold the session?
        // deserialize_expr_proto(pb::Expr::decode(bytes)?, &ExprRegistry::default())
        vortex_bail!("Expression array deserialization not yet implemented")
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array> {
        let input = children.get(0, &metadata.scope_dtype, len)?;

        Ok(ExpressionArray {
            expression: metadata.expression.clone(),
            dtype: dtype.clone(),
            input,
            stats: ArrayStats::default(),
        })
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        array.input = children
            .into_iter()
            .next()
            .vortex_expect("Expression array must have exactly one child");
        Ok(())
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        array
            .expression
            .evaluate(&array.input)?
            .execute::<Canonical>(ctx)
    }

    fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array)
    }

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ExpressionArray {
                expression: array.expression.clone(),
                dtype: array.dtype.clone(),
                input: array.input.slice(range),
                stats: Default::default(),
            }
            .into_array(),
        ))
    }
}

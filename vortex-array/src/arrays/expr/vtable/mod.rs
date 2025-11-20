// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod canonical;
mod operations;
pub mod operator;
mod visitor;

use std::fmt::Debug;

pub use operator::ExprOptimizationRule;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_vector::Vector;

use crate::arrays::expr::ExprArray;
use crate::execution::ExecutionCtx;
use crate::expr::Expression;
use crate::serde::ArrayChildren;
use crate::vtable::{NotSupported, VTable};
use crate::{Array, ArrayOperator, EncodingId, EncodingRef, vtable};

vtable!(Expr);

#[derive(Clone, Debug)]
pub struct ExprEncoding;

impl VTable for ExprVTable {
    type Array = ExprArray;
    type Encoding = ExprEncoding;
    type Metadata = ExprArrayMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = NotSupported;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type OperatorVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.expr")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ExprEncoding.as_ref())
    }

    fn metadata(array: &ExprArray) -> VortexResult<Self::Metadata> {
        Ok(ExprArrayMetadata((array.expr.clone(), array.dtype.clone())))
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(_bytes: &[u8]) -> VortexResult<Self::Metadata> {
        vortex_bail!("unsupported")
    }

    fn build(
        _encoding: &ExprEncoding,
        dtype: &DType,
        len: usize,
        ExprArrayMetadata((expr, root_dtype)): &Self::Metadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ExprArray> {
        if !buffers.is_empty() {
            vortex_bail!("Expected 0 buffers, got {}", buffers.len());
        }

        let Ok(child) = children.get(0, root_dtype, len) else {
            vortex_bail!("Expected 1 child, got {}", children.len());
        };

        ExprArray::try_new(child, expr.clone(), dtype.clone())
    }

    fn execute(array: &Self::Array, ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        let scope = array.child().execute_batch(ctx)?;
        array.expr().execute(&scope, array.child().dtype())
    }
}

pub struct ExprArrayMetadata((Expression, DType));

impl Debug for ExprArrayMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Since this is used in display method we can omit the dtype.
        self.0.0.fmt_sql(f)
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod canonical;
mod operations;
mod operator;
mod visitor;

use std::fmt::Debug;

use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::expr::ExprArray;
use crate::expr::Expression;
use crate::serde::ArrayChildren;
use crate::vtable::{NotSupported, VTable};
use crate::{EncodingId, EncodingRef, vtable};

vtable!(Expr);

#[derive(Clone, Debug)]
pub struct ExprEncoding;

impl VTable for ExprVTable {
    type Array = ExprArray;
    type Encoding = ExprEncoding;
    type Metadata = ExprDisplay;

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
        Ok(ExprDisplay(array.expr.clone()))
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
        metadata: &Self::Metadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ExprArray> {
        if !buffers.is_empty() {
            vortex_bail!("Expected 0 buffers, got {}", buffers.len());
        }

        let Ok(child) = children.get(0, dtype, len) else {
            vortex_bail!("Expected 1 child, got {}", children.len());
        };

        ExprArray::try_new(child, metadata.0.clone(), dtype.clone())
    }
}

pub struct ExprDisplay(Expression);

impl Debug for ExprDisplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt_sql(f)
    }
}

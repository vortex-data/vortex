// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Pushdown of `Eq` and `NotEq` against an OnPair column. We forward the
//! constant operand directly to `OnPairColumnView::equals`, which evaluates
//! the predicate on the compressed token stream without decoding rows.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::OnPair;

impl CompareKernel for OnPair {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !matches!(operator, CompareOperator::Eq | CompareOperator::NotEq) {
            return Ok(None);
        }
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        compare_eq_constant(lhs, &constant, operator)
    }
}

fn needle_bytes(scalar: &Scalar) -> Option<Vec<u8>> {
    match scalar.dtype() {
        DType::Utf8(_) => scalar.as_utf8().value().map(|s| s.as_bytes().to_vec()),
        DType::Binary(_) => scalar.as_binary().value().map(|b| b.to_vec()),
        _ => None,
    }
}

fn compare_eq_constant(
    lhs: ArrayView<'_, OnPair>,
    rhs: &Scalar,
    operator: CompareOperator,
) -> VortexResult<Option<ArrayRef>> {
    let Some(needle) = needle_bytes(rhs) else {
        return Ok(None);
    };

    let column = lhs.column()?;
    let raw = column
        .equals_bitmap(&needle)
        .map_err(|e| vortex_err!("OnPair equals pushdown failed: {e}"))?;
    let bool_buf = BitBuffer::new(ByteBuffer::from(raw), lhs.array().len());
    let bool_buf = if operator == CompareOperator::NotEq {
        !bool_buf
    } else {
        bool_buf
    };
    let nullability = lhs
        .array()
        .validity()?
        .union_nullability(rhs.dtype().nullability());
    Ok(Some(BoolArray::new(bool_buf, nullability).into_array()))
}

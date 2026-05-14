// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! `Eq` / `NotEq` against a constant. Each row's decoded bytes are streamed
//! through `DecodeView::for_each_dict_slice`, comparing prefix-wise against
//! the needle, so most non-matches short-circuit before any decode work.

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

use crate::OnPair;
use crate::decode::DecodeView;
use crate::decode::OwnedDecodeInputs;

impl CompareKernel for OnPair {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !matches!(operator, CompareOperator::Eq | CompareOperator::NotEq) {
            return Ok(None);
        }
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        let Some(needle) = needle_bytes(&constant) else {
            return Ok(None);
        };

        let inputs = OwnedDecodeInputs::collect(lhs, ctx)?;
        let dv = inputs.view();
        let n = lhs.array().len();
        let mut bytes = vec![0u8; n.div_ceil(8)];
        for row in 0..n {
            if row_equals_needle(&dv, row, &needle) {
                bytes[row / 8] |= 1u8 << (row % 8);
            }
        }
        let mut bool_buf = BitBuffer::new(ByteBuffer::from(bytes), n);
        if operator == CompareOperator::NotEq {
            bool_buf = !bool_buf;
        }
        let validity = lhs
            .array()
            .validity()?
            .union_nullability(constant.dtype().nullability());
        Ok(Some(BoolArray::new(bool_buf, validity).into_array()))
    }
}

fn needle_bytes(scalar: &Scalar) -> Option<Vec<u8>> {
    match scalar.dtype() {
        DType::Utf8(_) => scalar.as_utf8().value().map(|s| s.as_bytes().to_vec()),
        DType::Binary(_) => scalar.as_binary().value().map(|b| b.to_vec()),
        _ => None,
    }
}

/// True iff row `r` decodes to exactly `needle`.
fn row_equals_needle(dv: &DecodeView<'_>, r: usize, needle: &[u8]) -> bool {
    let mut pos = 0usize;
    let n = needle.len();
    let needle_ptr = needle.as_ptr();
    let ok = dv.for_each_dict_slice(r, |slice| {
        let take = slice.len();
        // Fast-path: bail on length overflow first so we never compare a
        // partial slice that would walk past `needle`.
        if pos + take > n {
            return false;
        }
        // SAFETY: `pos + take <= n`, `take == slice.len()`. Compares
        // `needle[pos..pos+take]` with `slice` via raw `memcmp`-style
        // pointer math. The branch on length above is the only check.
        let eq = unsafe {
            let lhs = needle_ptr.add(pos);
            std::slice::from_raw_parts(lhs, take) == slice
        };
        if !eq {
            return false;
        }
        pos += take;
        true
    });
    ok && pos == n
}

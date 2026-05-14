// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! `Eq` / `NotEq` against a constant via **token-aware** comparison.
//!
//! OnPair's compressor encodes every byte string deterministically via
//! greedy LPM against the same dictionary, so two byte strings are
//! equal **iff** their LPM token sequences are equal. We tokenise the
//! needle once and then compare the row's `codes[lo..hi]` slice
//! directly against the tokenised needle as `&[u16]` — no row decode.
//!
//! Edge case: if the needle contains a byte that has no dict entry at
//! all (degenerate dict; OnPair training normally guarantees every
//! single-byte token), no row can possibly equal the needle, since
//! every row was compressed against the same dict. We return an
//! all-zeros bitmap (or all-ones for `NotEq`).

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
use crate::decode::OwnedDecodeInputs;
use crate::lpm::DictIndex;
use crate::lpm::tokenize_needle;

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

        let index = DictIndex::build(&dv);
        if let Some(needle_toks) = tokenize_needle(&dv, &index, &needle) {
            let codes = dv.codes;
            let codes_offsets = dv.codes_offsets;
            for r in 0..n {
                let lo = codes_offsets[r] as usize;
                let hi = codes_offsets[r + 1] as usize;
                // SAFETY: codes_offsets validated at construction time.
                let row_toks = unsafe { codes.get_unchecked(lo..hi) };
                if row_toks == needle_toks.as_slice() {
                    bytes[r / 8] |= 1u8 << (r % 8);
                }
            }
        }
        // If `tokenize_needle` returned None, no row can equal the
        // needle (every row was compressed against the same dict, so
        // any byte not in the dict can't appear in any row either).
        // Leave the bitmap zeroed.

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

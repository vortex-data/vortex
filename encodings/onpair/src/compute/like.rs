// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Pattern matching. Three SQL `LIKE` shapes are accelerated by streaming
//! decoded dict slices and matching against the literal needle. Everything
//! else (escapes, wildcards in the middle, character classes, case-insensitive
//! matching) returns `None` and is handled by Vortex's default scalar path.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::decode::DecodeView;
use crate::decode::OwnedDecodeInputs;

#[derive(Debug)]
enum PatternShape<'a> {
    Equals(&'a [u8]),
    StartsWith(&'a [u8]),
    Contains(&'a [u8]),
}

fn classify(pattern: &[u8]) -> Option<PatternShape<'_>> {
    if pattern.contains(&b'_') || pattern.contains(&b'\\') {
        return None;
    }
    let first_pct = pattern.iter().position(|&b| b == b'%');
    let last_pct = pattern.iter().rposition(|&b| b == b'%');
    match (first_pct, last_pct) {
        (None, None) => Some(PatternShape::Equals(pattern)),
        (Some(0), Some(end)) if end == pattern.len() - 1 && pattern.len() >= 2 => {
            let inner = &pattern[1..pattern.len() - 1];
            if inner.contains(&b'%') {
                None
            } else {
                Some(PatternShape::Contains(inner))
            }
        }
        (Some(p), Some(q)) if p == q && q == pattern.len() - 1 => {
            Some(PatternShape::StartsWith(&pattern[..pattern.len() - 1]))
        }
        _ => None,
    }
}

impl LikeKernel for OnPair {
    fn like(
        array: ArrayView<'_, Self>,
        pattern: &ArrayRef,
        options: LikeOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if options.case_insensitive {
            return Ok(None);
        }
        let Some(scalar) = pattern.as_constant() else {
            return Ok(None);
        };
        let pattern_bytes: Vec<u8> = if let Some(s) = scalar.as_utf8_opt() {
            let Some(v) = s.value() else { return Ok(None) };
            v.as_bytes().to_vec()
        } else if let Some(b) = scalar.as_binary_opt() {
            let Some(v) = b.value() else { return Ok(None) };
            v.to_vec()
        } else {
            return Ok(None);
        };

        let Some(shape) = classify(&pattern_bytes) else {
            return Ok(None);
        };

        let inputs = OwnedDecodeInputs::collect(array, ctx)?;
        let dv = inputs.view();
        let n = array.array().len();
        let mut bytes = vec![0u8; n.div_ceil(8)];
        for row in 0..n {
            let matched = match &shape {
                PatternShape::Equals(needle) => row_equals(&dv, row, needle),
                PatternShape::StartsWith(prefix) => row_starts_with(&dv, row, prefix),
                PatternShape::Contains(sub) => row_contains(&dv, row, sub),
            };
            if matched {
                bytes[row / 8] |= 1u8 << (row % 8);
            }
        }
        let mut bool_buf = BitBuffer::new(ByteBuffer::from(bytes), n);
        if options.negated {
            bool_buf = !bool_buf;
        }
        let validity = array
            .array()
            .validity()?
            .union_nullability(scalar.dtype().nullability());
        Ok(Some(BoolArray::new(bool_buf, validity).into_array()))
    }
}

fn row_equals(dv: &DecodeView<'_>, r: usize, needle: &[u8]) -> bool {
    let mut pos = 0usize;
    let n = needle.len();
    let needle_ptr = needle.as_ptr();
    let ok = dv.for_each_dict_slice(r, |slice| {
        let take = slice.len();
        if pos + take > n {
            return false;
        }
        // SAFETY: `pos + take <= n`.
        let eq = unsafe { std::slice::from_raw_parts(needle_ptr.add(pos), take) == slice };
        if !eq {
            return false;
        }
        pos += take;
        true
    });
    ok && pos == n
}

fn row_starts_with(dv: &DecodeView<'_>, r: usize, prefix: &[u8]) -> bool {
    if prefix.is_empty() {
        return true;
    }
    let mut pos = 0usize;
    let mut matched = false;
    let plen = prefix.len();
    let prefix_ptr = prefix.as_ptr();
    dv.for_each_dict_slice(r, |slice| {
        let remaining = plen - pos;
        let take = slice.len().min(remaining);
        // SAFETY:
        // * `pos + take <= plen` because `take <= remaining`.
        // * `take <= slice.len()` by construction.
        let eq = unsafe {
            let lhs = std::slice::from_raw_parts(prefix_ptr.add(pos), take);
            let rhs = slice.get_unchecked(..take);
            lhs == rhs
        };
        if !eq {
            return false;
        }
        pos += take;
        if pos == plen {
            matched = true;
            return false; // short-circuit, prefix satisfied
        }
        true
    });
    matched
}

/// Substring match. We decode the row lazily into a scratch buffer and run
/// a byte-level scan; cheap for the small per-row strings OnPair targets.
fn row_contains(dv: &DecodeView<'_>, r: usize, sub: &[u8]) -> bool {
    if sub.is_empty() {
        return true;
    }
    let mut buf: Vec<u8> = Vec::with_capacity(64);
    dv.decode_row_into(r, &mut buf);
    buf.windows(sub.len()).any(|w| w == sub)
}

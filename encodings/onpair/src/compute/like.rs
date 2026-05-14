// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Pattern matching kernel. We recognise three SQL `LIKE` shapes and forward
//! them directly to OnPair's compressed-domain predicates:
//!
//! - `LIKE 'literal'`   -> `OnPairColumn::equals`
//! - `LIKE 'prefix%'`   -> `OnPairColumn::starts_with`
//! - `LIKE '%substr%'`  -> `OnPairColumn::contains`
//!
//! Anything else (escapes, mid-pattern wildcards, character classes, case
//! insensitivity) falls back to the default scalar implementation.

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
use vortex_error::vortex_err;

use crate::OnPair;

#[derive(Debug)]
enum PatternShape<'a> {
    Equals(&'a [u8]),
    StartsWith(&'a [u8]),
    Contains(&'a [u8]),
}

fn classify(pattern: &[u8]) -> Option<PatternShape<'_>> {
    // We do not handle escapes or character classes.
    if pattern.contains(&b'_') || pattern.contains(&b'\\') {
        return None;
    }
    let first_pct = pattern.iter().position(|&b| b == b'%');
    let last_pct = pattern.iter().rposition(|&b| b == b'%');
    match (first_pct, last_pct) {
        (None, None) => Some(PatternShape::Equals(pattern)),
        (Some(0), Some(end)) if end == pattern.len() - 1 && pattern.len() >= 2 => {
            // `%substr%`: the substring between the two anchors must be
            // wildcard-free.
            let inner = &pattern[1..pattern.len() - 1];
            if inner.contains(&b'%') {
                None
            } else {
                Some(PatternShape::Contains(inner))
            }
        }
        (Some(p), Some(q)) if p == q && q == pattern.len() - 1 => {
            // `prefix%`.
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
        _ctx: &mut ExecutionCtx,
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

        let column = array.column()?;
        let raw = match shape {
            PatternShape::Equals(s) => column.equals_bitmap(s),
            PatternShape::StartsWith(s) => column.starts_with_bitmap(s),
            PatternShape::Contains(s) => column.contains_bitmap(s),
        }
        .map_err(|e| vortex_err!("OnPair like pushdown failed: {e}"))?;

        let mut bool_buf = BitBuffer::new(ByteBuffer::from(raw), array.array().len());
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

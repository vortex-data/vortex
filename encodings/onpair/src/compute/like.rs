// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! `LIKE` pushdown for OnPair. Three pattern shapes are accelerated;
//! everything else returns `None` so the caller decompresses + runs the
//! scalar `LIKE` on the canonical bytes.
//!
//! * `'literal'` — token-aware equality. LPM-tokenise the literal once
//!   and compare the row's `codes[lo..hi]` against the tokenised needle
//!   as `&[u16]`. Full byte equality is exactly equivalent to full LPM
//!   token-sequence equality, so this is sound and skips row decode
//!   entirely.
//! * `'prefix%'` — byte-streaming via `DecodeView::for_each_dict_slice`
//!   with a single length check up front. The naive "tokenise the
//!   prefix and compare token prefix" trick is **wrong** because the
//!   LPM of the row's leading bytes may extend its last token past the
//!   literal prefix's tokenisation boundary. Streaming dict slices and
//!   comparing prefix-wise is the correct minimum-work option.
//! * `'%substring%'` — decode each row into a small reusable scratch
//!   buffer and run `memchr::memmem::Finder::find`, which is SIMD-
//!   accelerated (SSE2/AVX2 on x86_64, NEON on aarch64) and Two-Way
//!   underneath. The `Finder` is built once per kernel call and reused
//!   across every row.
//!
//! Escapes (`\\`), single-character wildcards (`_`), mid-pattern
//! wildcards, and `case_insensitive: true` all bail out with `None`.

use memchr::memmem;
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
use crate::lpm::DictIndex;
use crate::lpm::tokenize_needle;

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
        match shape {
            PatternShape::Equals(needle) => {
                let index = DictIndex::build(&dv);
                if let Some(needle_toks) = tokenize_needle(&dv, &index, needle) {
                    let codes = dv.codes;
                    let codes_offsets = dv.codes_offsets;
                    for r in 0..n {
                        let lo = codes_offsets[r] as usize;
                        let hi = codes_offsets[r + 1] as usize;
                        // SAFETY: codes_offsets validated at construction.
                        let row_toks = unsafe { codes.get_unchecked(lo..hi) };
                        if row_toks == needle_toks.as_slice() {
                            bytes[r / 8] |= 1u8 << (r % 8);
                        }
                    }
                }
                // Else: needle has a byte not in the dict, no row matches.
            }
            PatternShape::StartsWith(prefix) => {
                if prefix.is_empty() {
                    fill_all(&mut bytes, n);
                } else {
                    for r in 0..n {
                        if row_starts_with(&dv, r, prefix) {
                            bytes[r / 8] |= 1u8 << (r % 8);
                        }
                    }
                }
            }
            PatternShape::Contains(sub) => {
                if sub.is_empty() {
                    fill_all(&mut bytes, n);
                } else {
                    contains_into_bitmap(&dv, sub, n, &mut bytes);
                }
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

/// `LIKE 'prefix%'` — byte-stream the row's dict slices, comparing
/// against `prefix` and short-circuiting on the first mismatch or once
/// the prefix is satisfied.
fn row_starts_with(dv: &DecodeView<'_>, r: usize, prefix: &[u8]) -> bool {
    let mut pos = 0usize;
    let mut matched = false;
    let plen = prefix.len();
    let prefix_ptr = prefix.as_ptr();
    dv.for_each_dict_slice(r, |slice| {
        let remaining = plen - pos;
        let take = slice.len().min(remaining);
        // SAFETY: `pos + take <= plen` because `take <= remaining`,
        //         and `take <= slice.len()` by construction.
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

/// `%substring%` pushdown via SIMD-accelerated `memmem`. The `Finder`
/// is built once and reused across every row's decoded bytes; the
/// scratch buffer is reused too so each row decode reuses the same
/// allocation.
fn contains_into_bitmap(dv: &DecodeView<'_>, sub: &[u8], n: usize, out: &mut [u8]) {
    let finder = memmem::Finder::new(sub);
    let mut scratch: Vec<u8> = Vec::with_capacity(64);
    for r in 0..n {
        scratch.clear();
        dv.decode_row_into(r, &mut scratch);
        if finder.find(&scratch).is_some() {
            out[r / 8] |= 1u8 << (r % 8);
        }
    }
}

fn fill_all(bytes: &mut [u8], n: usize) {
    bytes.fill(0xff);
    if !n.is_multiple_of(8) {
        let last = n / 8;
        bytes[last] = (1u8 << (n % 8)) - 1;
    }
}

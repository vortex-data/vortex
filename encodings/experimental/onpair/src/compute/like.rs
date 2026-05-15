// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! `LIKE` pushdown for OnPair. Three pattern shapes are accelerated;
//! everything else returns `None` so the caller decompresses + runs the
//! scalar `LIKE` on the canonical bytes.
//!
//! * `'literal'` — token-aware equality (LPM-tokenise the literal once
//!   and compare the row's `codes[lo..hi]` against the tokenised needle
//!   as `&[u16]`). No row decode.
//! * `'prefix%'` — OnPair-style [`PrefixAutomaton`][crate::dfa::PrefixAutomaton]:
//!   tokenise the prefix and precompute valid-divergence intervals for
//!   each query position. Per-row scan is `≤ q + 1` `u16` comparisons
//!   plus one interval check; no decode at all in the hot path.
//! * `'%substring%'` — dict-bloom skip + `memchr::memmem` over the
//!   decoded row only when needed.
//!   [`ContainsBloom`][crate::dfa::ContainsBloom] precomputes "this
//!   dict entry contains the substring" and "some suffix of this entry
//!   could start a cross-token match". Most rows resolve via the bloom
//!   without touching `dict_bytes`; the rest fall through to a
//!   scratch-buffer decode + memmem.
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
use crate::dfa::ContainsBloom;
use crate::dfa::PrefixAutomaton;
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
                    let needle_slice = needle_toks.as_slice();
                    for r in 0..n {
                        let lo = codes_offsets[r] as usize;
                        let hi = codes_offsets[r + 1] as usize;
                        // SAFETY: codes_offsets validated at construction.
                        let row_toks = unsafe { codes.get_unchecked(lo..hi) };
                        if row_toks == needle_slice {
                            bytes[r / 8] |= 1u8 << (r % 8);
                        }
                    }
                }
                // Else: needle has a byte not in the dict ⇒ no row matches.
            }
            PatternShape::StartsWith(prefix) => {
                if prefix.is_empty() {
                    fill_all(&mut bytes, n);
                } else if let Some(automaton) = PrefixAutomaton::build(&dv, prefix) {
                    let codes = dv.codes;
                    let codes_offsets = dv.codes_offsets;
                    for r in 0..n {
                        let lo = codes_offsets[r] as usize;
                        let hi = codes_offsets[r + 1] as usize;
                        // SAFETY: codes_offsets validated at construction.
                        let row_toks = unsafe { codes.get_unchecked(lo..hi) };
                        if automaton.matches(row_toks) {
                            bytes[r / 8] |= 1u8 << (r % 8);
                        }
                    }
                }
                // Else: prefix has a byte not in the dict ⇒ no row matches.
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

/// `%substring%` pushdown: dict-bloom skip + per-row decode + memmem.
fn contains_into_bitmap(dv: &DecodeView<'_>, sub: &[u8], n: usize, out: &mut [u8]) {
    let bloom = ContainsBloom::build(dv, sub);
    let finder = memmem::Finder::new(sub);
    let mut scratch: Vec<u8> = Vec::with_capacity(64);
    let codes = dv.codes;
    let codes_offsets = dv.codes_offsets;
    for r in 0..n {
        let lo = codes_offsets[r] as usize;
        let hi = codes_offsets[r + 1] as usize;
        // SAFETY: codes_offsets validated at construction.
        let row_toks = unsafe { codes.get_unchecked(lo..hi) };
        let hit = match bloom.classify(row_toks) {
            Some(b) => b,
            None => {
                scratch.clear();
                dv.decode_row_into(r, &mut scratch);
                finder.find(&scratch).is_some()
            }
        };
        if hit {
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

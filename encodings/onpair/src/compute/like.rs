// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! `LIKE` pushdown for OnPair. Only the two **decode-free** shapes
//! `'literal'` (token equality) and `'prefix%'` (interval-checked
//! token-aware automaton) are pushed. `'%contains%'` falls through to
//! canonicalize + scalar `LIKE` — that path runs the bulk 4×-unrolled
//! decoder and a single SIMD `memmem` over the whole buffer, which
//! outperforms any per-row decode-then-search loop on long-string
//! corpora (verified on FineWeb NVMe q3/q6/q7).
//!
//! Escapes (`\\`), single-character wildcards (`_`), mid-pattern
//! wildcards, and `case_insensitive: true` all bail out with `None`.

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
use crate::decode::OwnedDecodeInputs;
use crate::dfa::PrefixAutomaton;
use crate::lpm::DictIndex;
use crate::lpm::tokenize_needle;

#[derive(Debug)]
enum PatternShape<'a> {
    Equals(&'a [u8]),
    StartsWith(&'a [u8]),
}

/// Recognise the LIKE pattern shapes OnPair can resolve **without
/// decoding the row**:
///
/// * `'literal'`  — exact equality. LPM-tokenise once, compare `&[u16]`.
/// * `'prefix%'`  — `PrefixAutomaton` (interval check per row token).
///
/// `'%contains%'` deliberately returns `None`: bench on FineWeb NVMe
/// (q3/q6/q7) showed the per-row "decode + memmem" pushdown is ~2×
/// slower than canonicalize + scalar `LIKE`, because canonical decode
/// hits the 4×-unrolled bulk decode loop and the scalar `LIKE` runs a
/// single SIMD `memmem` over the whole buffer. Falling through is the
/// minimum-work option for contains.
fn classify(pattern: &[u8]) -> Option<PatternShape<'_>> {
    if pattern.contains(&b'_') || pattern.contains(&b'\\') {
        return None;
    }
    let first_pct = pattern.iter().position(|&b| b == b'%');
    let last_pct = pattern.iter().rposition(|&b| b == b'%');
    match (first_pct, last_pct) {
        (None, None) => Some(PatternShape::Equals(pattern)),
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

fn fill_all(bytes: &mut [u8], n: usize) {
    bytes.fill(0xff);
    if !n.is_multiple_of(8) {
        let last = n / 8;
        bytes[last] = (1u8 << (n % 8)) - 1;
    }
}

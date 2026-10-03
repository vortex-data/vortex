// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare against a constant on the bin ids.
//!
//! A bin covers a contiguous range of values, so for most bins a comparison has the same answer
//! for every value in it. Blocks whose ids all fall in such bins are answered from their decoded
//! ids alone, skipping the offsets; only blocks holding a value of a straddling bin are merged.
//! This needs the values themselves, so arrays coding differences (`lag > 0`) are left to the
//! generic path.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::EntropyBins;
use crate::EntropyBinsChunk;
use crate::array::EntropyBinsData;
use crate::array::Wide;
use crate::coder::CHUNK_VALUES;
use crate::decode::IDS_SCRATCH;
use crate::decode::OutInt;
use crate::decode::merge_block;
use crate::decode::parse_block;

const SIGN: u64 = 1 << 63;

/// The answer for every value of a bin.
const FALSE: u8 = 0;
const TRUE: u8 = 1;
const STRADDLES: u8 = 2;

impl CompareKernel for EntropyBins {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let data = lhs.data();
        if data.metadata.lag != 0 {
            return Ok(None);
        }
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        let Some(constant) = constant.as_primitive_opt() else {
            return Ok(None);
        };
        if constant.ptype() != data.ptype() {
            return Ok(None);
        }
        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let validity = lhs.validity()?.union_nullability(nullability);
        let bits = match_each_integer_ptype!(data.ptype(), |T| {
            let Some(c) = constant.typed_value::<T>() else {
                return Ok(None);
            };
            match compare_typed::<T>(data, c, operator)? {
                Some(bits) => bits,
                None => return Ok(None),
            }
        });
        Ok(Some(BoolArray::new(bits, validity).into_array()))
    }
}

/// Order-preserving latent of a value (matches the encoder's lag-0 latents).
fn latent<T: Wide>(v: T, signed: bool) -> u64 {
    if signed { v.wide() ^ SIGN } else { v.wide() }
}

fn holds(operator: CompareOperator, l: u64, c: u64) -> bool {
    match operator {
        CompareOperator::Eq => l == c,
        CompareOperator::NotEq => l != c,
        CompareOperator::Lt => l < c,
        CompareOperator::Lte => l <= c,
        CompareOperator::Gt => l > c,
        CompareOperator::Gte => l >= c,
    }
}

/// The expected share of a chunk's values that fall in straddling bins, from the bin weights.
fn expected_straddlers(chunk: &EntropyBinsChunk, class: &[u8]) -> f64 {
    let total: f64 = chunk.weights.iter().map(|&w| f64::from(w)).sum();
    let straddling: f64 = chunk
        .weights
        .iter()
        .zip(class)
        .filter(|&(_, &c)| c == STRADDLES)
        .map(|(&w, _)| f64::from(w))
        .sum();
    if total > 0.0 { straddling / total } else { 1.0 }
}

/// The answer for each bin of `chunk` (zero-padded to 64 entries for the vector lookup).
fn classes(chunk: &EntropyBinsChunk, operator: CompareOperator, c: u64) -> Vec<u8> {
    let mut out = vec![FALSE; chunk.lowers.len().max(64)];
    for (class, (&lo, &w)) in out.iter_mut().zip(chunk.lowers.iter().zip(&chunk.widths)) {
        let hi = if w >= 64 {
            u64::MAX
        } else {
            lo.saturating_add((1u64 << w) - 1)
        };
        let (at_lo, at_hi) = (holds(operator, lo, c), holds(operator, hi, c));
        // Order comparisons change answer at most once over a range; equality only at `c`.
        let uniform = match operator {
            CompareOperator::Eq | CompareOperator::NotEq => lo == hi || c < lo || c > hi,
            _ => at_lo == at_hi,
        };
        *class = if !uniform {
            STRADDLES
        } else if at_lo {
            TRUE
        } else {
            FALSE
        };
    }
    out
}

fn compare_typed<T: NativePType + OutInt + Wide>(
    data: &EntropyBinsData,
    rhs: T,
    operator: CompareOperator,
) -> VortexResult<Option<vortex_buffer::BitBuffer>> {
    let signed = data.ptype().is_signed_int();
    let rhs_latent = latent(rhs, signed);
    let (start, stop) = data.slice_range();
    let len = stop - start;
    if len == 0 {
        return Ok(Some(BitBufferMut::new_unset(0).freeze()));
    }
    let bv = data.block_values();
    let n_rows = data.unsliced_rows();
    let first = start / bv;
    let last = (stop - 1) / bv;
    let blocks_per_chunk = CHUNK_VALUES / bv;
    // Where most blocks hold a straddling value, classifying first only adds work. If that holds
    // for every chunk, decoding and comparing the values is as fast as this kernel can be.
    let straddle_heavy = |ci: usize| {
        let chunk = &data.metadata.chunks[ci];
        expected_straddlers(chunk, &classes(chunk, operator, rhs_latent)) * bv as f64 >= 1.0
    };
    if (first / blocks_per_chunk..=last / blocks_per_chunk).all(straddle_heavy) {
        return Ok(None);
    }
    let words_per_block = bv / 64;
    let mut words = BufferMut::<u64>::zeroed((last + 1 - first) * words_per_block);
    let mut ids = vec![0u8; IDS_SCRATCH];
    let mut values = vec![T::default(); bv];
    let bytes = data.data.as_slice();
    let mut b = first;
    while b <= last {
        let ci = b / blocks_per_chunk;
        let chunk_last = ((ci + 1) * blocks_per_chunk - 1).min(last);
        let chunk = &data.metadata.chunks[ci];
        let class = classes(chunk, operator, rhs_latent);
        let seg =
            &mut words[(b - first) * words_per_block..(chunk_last + 1 - first) * words_per_block];
        // A straddle-heavy chunk next to lighter ones: decode its blocks in bulk.
        if expected_straddlers(chunk, &class) * bv as f64 >= 1.0 {
            let stop_row = ((chunk_last + 1) * bv).min(n_rows);
            let decoded = data.decode_range::<T>(b * bv, stop_row)?;
            compare_values(seg, &decoded, rhs, operator);
            b = chunk_last + 1;
            continue;
        }
        let decoder = data.decoder(ci)?;
        for (k, out) in seg.chunks_mut(words_per_block).enumerate() {
            let block = b + k;
            let n = bv.min(n_rows - block * bv);
            let view = parse_block(bytes, data.block_start(block), n, decoder.table.as_ref())?;
            if let Some(u) = view.uniform {
                match class[usize::from(u)] {
                    TRUE => fill_true(out, n),
                    FALSE => {}
                    _ => {
                        decoder.ids(&view, &mut ids, usize::MAX);
                        merge_block(decoder, &view, &ids, &mut values, &[]);
                        compare_values(out, &values[..n], rhs, operator);
                    }
                }
                continue;
            }
            decoder.ids(&view, &mut ids, usize::MAX);
            if !classify(&ids[..n], &class, out) {
                merge_block(decoder, &view, &ids, &mut values, &[]);
                compare_values(out, &values[..n], rhs, operator);
            }
        }
        b = chunk_last + 1;
    }
    let offset = start - first * bv;
    Ok(Some(
        BitBufferMut::from_buffer(words.into_byte_buffer(), offset, len).freeze(),
    ))
}

fn fill_true(out: &mut [u64], n: usize) {
    for (i, w) in out.iter_mut().enumerate() {
        let lo = i * 64;
        *w = if lo + 64 <= n {
            u64::MAX
        } else if lo < n {
            (1u64 << (n - lo)) - 1
        } else {
            0
        };
    }
}

/// Overwrite `out` with `values[i] <op> rhs`, one bit per value.
fn compare_values<T: NativePType>(
    out: &mut [u64],
    values: &[T],
    rhs: T,
    operator: CompareOperator,
) {
    match operator {
        CompareOperator::Eq => pack(out, values, |v| v.is_eq(rhs)),
        CompareOperator::NotEq => pack(out, values, |v| !v.is_eq(rhs)),
        CompareOperator::Lt => pack(out, values, |v| v.is_lt(rhs)),
        CompareOperator::Lte => pack(out, values, |v| v.is_le(rhs)),
        CompareOperator::Gt => pack(out, values, |v| v.is_gt(rhs)),
        CompareOperator::Gte => pack(out, values, |v| v.is_ge(rhs)),
    }
}

/// Pack a predicate over `values` into bits, 64 values per word, in a loop the compiler
/// vectorizes.
fn pack<T: Copy>(out: &mut [u64], values: &[T], pred: impl Fn(T) -> bool) {
    for (word, chunk) in out.iter_mut().zip(values.chunks(64)) {
        let mut w = 0u64;
        for (j, &v) in chunk.iter().enumerate() {
            w |= u64::from(pred(v)) << j;
        }
        *word = w;
    }
}

/// Write the answer bits of `ids` from per-bin classes. Returns `false` (leaving `out` partly
/// written) if some id's bin straddles the constant.
fn classify(ids: &[u8], class: &[u8], out: &mut [u64]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if class.len() <= 64 && crate::x86::has_avx512() {
        // SAFETY: AVX-512 is available, `class` has 64 entries, `out` has a word per 64 ids.
        return unsafe { crate::x86::classify_ids(ids, class, out) };
    }
    for (i, &id) in ids.iter().enumerate() {
        match class[usize::from(id)] {
            TRUE => out[i / 64] |= 1 << (i % 64),
            FALSE => {}
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::scalar_fn::fns::binary::CompareKernel;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use crate::BLOCK_VALUES;
    use crate::EntropyBins;
    use crate::MAX_BLOCK_VALUES;

    const OPERATORS: [CompareOperator; 6] = [
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Lt,
        CompareOperator::Lte,
        CompareOperator::Gt,
        CompareOperator::Gte,
    ];

    /// Clustered values with rare outliers, so bins range from single values to wide spans.
    fn values(n: usize) -> Vec<i32> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        (0..n)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let base = i32::try_from(i / 3000).unwrap_or(0) * 7;
                match state % 50 {
                    0 => i32::try_from(state >> 40).unwrap_or(0) - 8_000_000,
                    1..10 => base + i32::try_from(state >> 54).unwrap_or(0),
                    _ => base + i32::try_from(state >> 61).unwrap_or(0),
                }
            })
            .collect()
    }

    #[rstest]
    #[case(BLOCK_VALUES, false)]
    #[case(MAX_BLOCK_VALUES, false)]
    #[case(BLOCK_VALUES, true)]
    fn matches_primitive(#[case] block_values: usize, #[case] scalar: bool) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let n = 20_000;
        let v = values(n);
        let validity = Validity::from_iter((0..n).map(|i| i % 13 != 0));
        let prim = PrimitiveArray::new(Buffer::from(v.clone()), validity);
        let encoded = EntropyBins::from_primitive(
            prim.as_view(),
            8,
            crate::EntropyBinsOptions::new(0, block_values),
        )?
        .into_array();
        #[cfg(target_arch = "x86_64")]
        crate::x86::set_force_scalar(scalar);
        let mut sorted = v;
        sorted.sort_unstable();
        let constants = [
            i32::MIN,
            sorted[0] - 1,
            sorted[0],
            sorted[n / 100],
            sorted[n / 2],
            sorted[n / 2] + 1,
            sorted[n - 1],
            i32::MAX,
        ];
        let (mut engaged, mut total) = (0, 0);
        for (a, b) in [(0, n), (1500, 9999), (4096, 8192), (n - 3, n)] {
            let lhs = encoded.slice(a..b)?;
            let lhs_view = lhs
                .as_opt::<EntropyBins>()
                .ok_or_else(|| vortex_error::vortex_err!("slice is not entropy bins"))?;
            let expected_lhs = prim.clone().into_array().slice(a..b)?;
            for &c in &constants {
                let rhs = ConstantArray::new(c, b - a).into_array();
                for op in OPERATORS {
                    total += 1;
                    let Some(got) =
                        <EntropyBins as CompareKernel>::compare(lhs_view, &rhs, op, &mut ctx)?
                    else {
                        continue;
                    };
                    engaged += 1;
                    let got = got.execute::<BoolArray>(&mut ctx)?;
                    let want = expected_lhs
                        .clone()
                        .binary(rhs.clone(), Operator::from(op))?
                        .execute::<BoolArray>(&mut ctx)?;
                    assert_arrays_eq!(got, want, &mut ctx);
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        crate::x86::set_force_scalar(false);
        // The data is deliberately straddle-rich; the kernel declines where it would not help.
        assert!(
            engaged * 4 >= total,
            "kernel engaged {engaged} of {total} times"
        );
        Ok(())
    }

    /// A straddle-heavy chunk (uniform values) next to one of a few distinct values: the kernel
    /// engages and decodes the heavy chunk in bulk.
    #[test]
    fn mixed_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut v: Vec<i32> = (0..crate::CHUNK_VALUES)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                i32::try_from(state >> 40).unwrap_or(0)
            })
            .collect();
        v.extend((0..9000).map(|i| (i % 7) * 1000));
        let n = v.len();
        let prim = PrimitiveArray::new(Buffer::from(v), Validity::NonNullable);
        let encoded = EntropyBins::from_primitive(
            prim.as_view(),
            8,
            crate::EntropyBinsOptions::new(0, BLOCK_VALUES),
        )?;
        let rhs = ConstantArray::new(1i32 << 23, n).into_array();
        for op in OPERATORS {
            let got =
                <EntropyBins as CompareKernel>::compare(encoded.as_view(), &rhs, op, &mut ctx)?
                    .ok_or_else(|| vortex_error::vortex_err!("kernel declined"))?
                    .execute::<BoolArray>(&mut ctx)?;
            let want = prim
                .clone()
                .into_array()
                .binary(rhs.clone(), Operator::from(op))?
                .execute::<BoolArray>(&mut ctx)?;
            assert_arrays_eq!(got, want, &mut ctx);
        }
        Ok(())
    }

    /// Uniform values fall in a few wide bins, so the constant straddles a heavy bin and the
    /// kernel leaves the comparison to the generic path.
    #[test]
    fn straddle_heavy_chunk() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let v: Vec<u32> = (0..9000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                u32::try_from(state >> 40).unwrap_or(0)
            })
            .collect();
        let prim = PrimitiveArray::new(Buffer::from(v), Validity::NonNullable);
        let encoded = EntropyBins::from_primitive(
            prim.as_view(),
            8,
            crate::EntropyBinsOptions::new(0, BLOCK_VALUES),
        )?;
        let chunk = &encoded.data().metadata.chunks[0];
        let rhs_value = 1u32 << 23;
        let class = super::classes(chunk, CompareOperator::Lt, u64::from(rhs_value));
        assert!(super::expected_straddlers(chunk, &class) * BLOCK_VALUES as f64 >= 1.0);
        let rhs = ConstantArray::new(rhs_value, 9000).into_array();
        let declined = <EntropyBins as CompareKernel>::compare(
            encoded.as_view(),
            &rhs,
            CompareOperator::Lt,
            &mut ctx,
        )?;
        assert!(declined.is_none());
        for op in OPERATORS {
            let got = encoded
                .clone()
                .into_array()
                .binary(rhs.clone(), Operator::from(op))?
                .execute::<BoolArray>(&mut ctx)?;
            let want = prim
                .clone()
                .into_array()
                .binary(rhs.clone(), Operator::from(op))?
                .execute::<BoolArray>(&mut ctx)?;
            assert_arrays_eq!(got, want, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn declines_differences() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let prim = PrimitiveArray::new(Buffer::from(values(5000)), Validity::NonNullable);
        let encoded = EntropyBins::from_primitive(
            prim.as_view(),
            8,
            crate::EntropyBinsOptions::new(1, BLOCK_VALUES),
        )?;
        let rhs = ConstantArray::new(7i32, 5000).into_array();
        let got = <EntropyBins as CompareKernel>::compare(
            encoded.as_view(),
            &rhs,
            CompareOperator::Lt,
            &mut ctx,
        )?;
        assert!(got.is_none());
        Ok(())
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Property tests for decimal byte-parts round trips, ops, serde, and sign extension.

use std::sync::LazyLock;

use hegel::TestCase;
use hegel::generators as gs;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::compute::conformance::hegel::EncodingCheckOptions;
use vortex_array::compute::conformance::hegel::check_encoding_roundtrip_and_ops;
use vortex_array::compute::conformance::hegel::draw_validity_for;
use vortex_array::compute::conformance::hegel::serialize_and_decode;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::i256;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::DecimalByteParts;
use super::DecimalBytePartsArray;
use super::DecimalBytePartsArraySlotsExt;
use super::testing::widen_msp;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

/// Largest magnitude a `Decimal(18, _)` can hold: 18 nines.
const MAX_I64: i64 = 10i64.pow(18) - 1;

/// Largest magnitude a `Decimal(38, _)` can hold: 38 nines.
const MAX_I128: i128 = 10i128.pow(38) - 1;

/// Bound on the high `i128` half of an `i256` draw. `10^37 * 2^128` is about `3.4e75`, so any
/// value built from it stays inside the 76 digits a `Decimal(76, _)` can hold.
const MAX_I256_HIGH: i128 = 10i128.pow(37);

/// Safe symmetric MSP bound for 1 lower part at `Decimal(38, _)`:
/// `(|msp| + 1) * 2^64 <= 10^38 - 1`, so any `u64` lower part stays within precision 38.
const MAX_MSP_ONE_LOWER_P38: i64 = 5_421_010_862_427_522_169;

/// Safe symmetric MSP bound for 3 lower parts at `Decimal(76, _)`:
/// `(|msp| + 1) * 2^192 <= 10^76 - 1`, so any three `u64` lower parts stay within precision 76.
const MAX_MSP_THREE_LOWER_P76: i64 = 1_593_091_911_132_452_276;

/// Rows per generated array. Small enough to shrink usefully, large enough that a chunked or
/// vectorized path is not trivially degenerate.
const MAX_LEN: usize = 48;

fn draw_len(tc: &TestCase) -> usize {
    match tc.draw(gs::integers::<u8>().max_value(5)) {
        0 => 0,
        1 => 1,
        _ => tc.draw(gs::integers::<usize>().min_value(1).max_value(MAX_LEN)),
    }
}

fn draw_nullability(tc: &TestCase) -> Nullability {
    if tc.draw(gs::booleans()) {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    }
}

fn draw_i64_decimal(tc: &TestCase, ctx: &mut ExecutionCtx) -> DecimalArray {
    let len = draw_len(tc);
    let values: Vec<i64> = tc.draw(
        gs::vecs(gs::integers::<i64>().min_value(-MAX_I64).max_value(MAX_I64))
            .min_size(len)
            .max_size(len),
    );
    let validity = draw_validity_for(tc, draw_nullability(tc), len);
    DecimalArray::new(
        Buffer::from(values),
        DecimalDType::new(18, 2),
        validity,
        ctx,
    )
}

/// An `i128`-backed decimal. The bounds keep values inside `Decimal(38, 2)` while still
/// reaching both sides of the 64-bit word boundary the encoding splits on.
fn draw_i128_decimal(tc: &TestCase, ctx: &mut ExecutionCtx) -> DecimalArray {
    let len = draw_len(tc);
    let values: Vec<i128> = tc.draw(
        gs::vecs(
            gs::integers::<i128>()
                .min_value(-MAX_I128)
                .max_value(MAX_I128),
        )
        .min_size(len)
        .max_size(len),
    );
    let validity = draw_validity_for(tc, draw_nullability(tc), len);
    DecimalArray::new(
        Buffer::from(values),
        DecimalDType::new(38, 2),
        validity,
        ctx,
    )
}

/// An `i256`-backed decimal, built from a signed high half and an unsigned low half so the
/// draw covers sign extension above the most significant part.
fn draw_i256_decimal(tc: &TestCase, ctx: &mut ExecutionCtx) -> DecimalArray {
    let len = draw_len(tc);
    let halves: Vec<(i128, u128)> = tc.draw(
        gs::vecs(gs::tuples2(
            gs::integers::<i128>()
                .min_value(-MAX_I256_HIGH)
                .max_value(MAX_I256_HIGH),
            gs::integers::<u128>(),
        ))
        .min_size(len)
        .max_size(len),
    );
    let values: Vec<i256> = halves
        .into_iter()
        .map(|(high, low)| i256::from_parts(low, high))
        .collect();
    let validity = draw_validity_for(tc, draw_nullability(tc), len);
    DecimalArray::new(
        Buffer::from(values),
        DecimalDType::new(76, 2),
        validity,
        ctx,
    )
}

fn draw_decimal(tc: &TestCase, ctx: &mut ExecutionCtx) -> DecimalArray {
    match tc.draw(gs::integers::<u8>().max_value(2)) {
        0 => draw_i64_decimal(tc, ctx),
        1 => draw_i128_decimal(tc, ctx),
        _ => draw_i256_decimal(tc, ctx),
    }
}

/// Independent two's-complement byte-level oracle for a decimal assembled from a signed `msp` and
/// `lower` 64-bit words (ordered most significant first).
///
/// Sign-fills every byte above the MSP directly rather than using the encoding's shift-and-or
/// assembly loop, so a missing sign extension above an MSP below the top word cannot cancel out.
fn oracle_le_bytes(msp: i64, lower: &[u64]) -> [u8; 32] {
    let sign_byte = if msp < 0 { 0xFFu8 } else { 0x00u8 };
    let mut bytes = [sign_byte; 32];
    let k = lower.len();
    for (i, &word) in lower.iter().enumerate() {
        let word_pos = k - 1 - i;
        bytes[word_pos * 8..(word_pos + 1) * 8].copy_from_slice(&word.to_le_bytes());
    }
    bytes[k * 8..(k + 1) * 8].copy_from_slice(&msp.to_le_bytes());
    bytes
}

fn oracle_decimal_array(
    msp: &[i64],
    lower_words: &[Vec<u64>],
    validity: Validity,
    decimal_dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) -> DecimalArray {
    let len = msp.len();
    let mut row_lower = [0u64; 3];
    let k = lower_words.len();

    match decimal_dtype.canonical_values_type() {
        DecimalType::I64 => {
            let values: Vec<i64> = (0..len)
                .map(|row| {
                    for (part_idx, part) in lower_words.iter().enumerate() {
                        row_lower[part_idx] = part[row];
                    }
                    let bytes = oracle_le_bytes(msp[row], &row_lower[..k]);
                    i64::from_le_bytes(bytes[..8].try_into().ok().vortex_expect("8 bytes"))
                })
                .collect();
            DecimalArray::new(Buffer::from(values), decimal_dtype, validity, ctx)
        }
        DecimalType::I128 => {
            let values: Vec<i128> = (0..len)
                .map(|row| {
                    for (part_idx, part) in lower_words.iter().enumerate() {
                        row_lower[part_idx] = part[row];
                    }
                    let bytes = oracle_le_bytes(msp[row], &row_lower[..k]);
                    i128::from_le_bytes(bytes[..16].try_into().ok().vortex_expect("16 bytes"))
                })
                .collect();
            DecimalArray::new(Buffer::from(values), decimal_dtype, validity, ctx)
        }
        DecimalType::I256 => {
            let values: Vec<i256> = (0..len)
                .map(|row| {
                    for (part_idx, part) in lower_words.iter().enumerate() {
                        row_lower[part_idx] = part[row];
                    }
                    let bytes = oracle_le_bytes(msp[row], &row_lower[..k]);
                    i256::from_le_bytes(bytes)
                })
                .collect();
            DecimalArray::new(Buffer::from(values), decimal_dtype, validity, ctx)
        }
        other => unreachable!("draw_encoded only selects I64/I128/I256 precisions, got {other:?}"),
    }
}

/// Draws a byte-parts array directly from parts alongside its independent canonical oracle.
///
/// Covers shapes `split_decimal` never emits — including 2 lower parts, MSPs below the top word
/// (0 lower parts at precision 19/39, 1 lower part at precision 39, 2 lower parts at precision
/// 76), and `Widen`-wrapped narrow MSPs (`i8`/`i16`/`i32`) — while keeping every assembled value
/// within the declared decimal precision.
fn draw_encoded_with_oracle(
    tc: &TestCase,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(DecimalBytePartsArray, DecimalArray)> {
    // (lower_part_count, precision, symmetric MSP bound)
    let (lower_part_count, precision, msp_bound): (usize, u8, Option<i64>) =
        match tc.draw(gs::integers::<u8>().max_value(6)) {
            0 => (0, 18, Some(MAX_I64)),
            1 => (0, 19, None),
            2 => (0, 39, None),
            3 => (1, 38, Some(MAX_MSP_ONE_LOWER_P38)),
            4 => (1, 39, None),
            5 => (2, 76, None),
            _ => (3, 76, Some(MAX_MSP_THREE_LOWER_P76)),
        };

    let decimal_dtype = DecimalDType::new(precision, 2);
    let len = draw_len(tc);
    let validity = draw_validity_for(tc, draw_nullability(tc), len);

    // Draw either a full `i64` MSP or a `Widen`-wrapped narrower signed MSP.
    let (msp_values, msp_array): (Vec<i64>, ArrayRef) = match tc
        .draw(gs::integers::<u8>().max_value(4))
    {
        0 => {
            let raw: Vec<i8> = tc.draw(gs::vecs(gs::integers::<i8>()).min_size(len).max_size(len));
            let msp_i64 = raw.iter().map(|&v| i64::from(v)).collect();
            let child = PrimitiveArray::new(Buffer::from(raw), validity.clone()).into_array();
            (msp_i64, widen_msp(child, decimal_dtype)?)
        }
        1 => {
            let raw: Vec<i16> =
                tc.draw(gs::vecs(gs::integers::<i16>()).min_size(len).max_size(len));
            let msp_i64 = raw.iter().map(|&v| i64::from(v)).collect();
            let child = PrimitiveArray::new(Buffer::from(raw), validity.clone()).into_array();
            (msp_i64, widen_msp(child, decimal_dtype)?)
        }
        2 => {
            let raw: Vec<i32> =
                tc.draw(gs::vecs(gs::integers::<i32>()).min_size(len).max_size(len));
            let msp_i64 = raw.iter().map(|&v| i64::from(v)).collect();
            let child = PrimitiveArray::new(Buffer::from(raw), validity.clone()).into_array();
            (msp_i64, widen_msp(child, decimal_dtype)?)
        }
        _ => {
            let raw: Vec<i64> = match msp_bound {
                Some(bound) => tc.draw(
                    gs::vecs(gs::integers::<i64>().min_value(-bound).max_value(bound))
                        .min_size(len)
                        .max_size(len),
                ),
                None => tc.draw(gs::vecs(gs::integers::<i64>()).min_size(len).max_size(len)),
            };
            let array =
                PrimitiveArray::new(Buffer::from(raw.clone()), validity.clone()).into_array();
            (raw, array)
        }
    };

    let lower_words: Vec<Vec<u64>> = (0..lower_part_count)
        .map(|_| tc.draw(gs::vecs(gs::integers::<u64>()).min_size(len).max_size(len)))
        .collect();
    let lower_arrays: Vec<ArrayRef> = lower_words
        .iter()
        .map(|part| {
            PrimitiveArray::new(Buffer::from(part.clone()), Validity::NonNullable).into_array()
        })
        .collect();

    let oracle = oracle_decimal_array(&msp_values, &lower_words, validity, decimal_dtype, ctx);
    let encoded =
        DecimalByteParts::try_new_with_lower_parts(msp_array, lower_arrays, decimal_dtype)?;
    Ok((encoded, oracle))
}

/// Encoding a decimal and decoding it must preserve values, storage width, scalar access, serde,
/// comparisons, and chained `slice`/`take`/`filter`/`mask` operations.
#[hegel::test(test_cases = 100, print_blob = true)]
fn decoded_survives_encode_then_decode(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal = draw_decimal(&tc, &mut ctx);

    let encoded = DecimalByteParts::encode(&decimal, &mut ctx).vortex_expect("encode");
    let encoded_ref = encoded.into_array();
    let round_tripped = encoded_ref
        .clone()
        .execute::<DecimalArray>(&mut ctx)
        .vortex_expect("execute");

    assert_eq!(round_tripped.values_type(), decimal.values_type());
    check_encoding_roundtrip_and_ops(
        &tc,
        &encoded_ref,
        &decimal.into_array(),
        &EncodingCheckOptions::default(),
        &mut ctx,
    )
    .vortex_expect("encode roundtrip and ops");
}

/// Directly constructed byte-parts arrays (including 2-part layouts, narrow `Widen` MSPs, and
/// sign extension above an MSP below the top word) must match an independent byte-level oracle
/// across bulk decode, `scalar_at`, serde metadata preservation, re-encoding, `compare`, and
/// chained `slice`/`take`/`filter`/`mask` operations.
#[hegel::test(test_cases = 100, print_blob = true)]
fn test_encoded_parts_match_oracle_and_ops(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let (encoded, oracle) = draw_encoded_with_oracle(&tc, &mut ctx).vortex_expect("draw encoded");
    let expected_lower_parts = encoded.lower_parts().len();
    let encoded_ref = encoded.into_array();
    let oracle_ref = oracle.clone().into_array();

    // Verify wire metadata preserves the exact lower-part count across v1 (0) and v2 (1..=3).
    let serde_decoded = serialize_and_decode(&encoded_ref, ctx.session()).vortex_expect("serde");
    assert_eq!(
        serde_decoded
            .as_opt::<DecimalByteParts>()
            .vortex_expect("decoded DecimalByteParts")
            .lower_parts()
            .len(),
        expected_lower_parts,
        "serde must preserve lower-part count"
    );

    check_encoding_roundtrip_and_ops(
        &tc,
        &encoded_ref,
        &oracle_ref,
        &EncodingCheckOptions::default(),
        &mut ctx,
    )
    .vortex_expect("encoded parts oracle and ops");

    let decoded = encoded_ref
        .execute::<DecimalArray>(&mut ctx)
        .vortex_expect("decode");
    let re_encoded = DecimalByteParts::encode(&decoded, &mut ctx).vortex_expect("re-encode");
    assert_arrays_eq!(re_encoded, oracle, &mut ctx);
}

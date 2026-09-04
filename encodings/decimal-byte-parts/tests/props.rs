// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Property tests for splitting decimals into byte parts and putting them back together.
//!
//! Every property here is the same shape: whatever the encoding does must be indistinguishable
//! from doing it to the canonical `DecimalArray`. Round tripping covers the split/assemble
//! pair directly; the compute properties cover it indirectly, since each one canonicalizes an
//! encoded array at the end.
//!
//! The generators deliberately reach the cases hand-written tests tend to miss: values that
//! straddle a 64-bit word boundary, negative values whose sign extension fills the words above
//! the most significant part, and null rows whose lower parts hold arbitrary bits.

#![expect(clippy::tests_outside_test_module)]

use hegel::TestCase;
use hegel::generators as gs;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::i256;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::DecimalBytePartsArray;
use vortex_decimal_byte_parts::split_decimal;
use vortex_error::VortexExpect;

/// Largest magnitude a `Decimal(38, _)` can hold: 38 nines.
const MAX_I128: i128 = 10i128.pow(38) - 1;

/// Bound on the high `i128` half of an `i256` draw. `10^37 * 2^128` is about `3.4e75`, so any
/// value built from it stays inside the 76 digits a `Decimal(76, _)` can hold.
const MAX_I256_HIGH: i128 = 10i128.pow(37);

/// Rows per generated array. Small enough to shrink usefully, large enough that a chunked or
/// vectorized path is not trivially degenerate.
const MAX_LEN: usize = 48;

fn ctx() -> ExecutionCtx {
    let session = array_session();
    vortex_decimal_byte_parts::initialize(&session);
    session.create_execution_ctx()
}

/// Encode a canonical decimal as byte parts, splitting wide values into lower parts.
fn encode(decimal: &DecimalArray) -> DecimalBytePartsArray {
    let parts = split_decimal(decimal).vortex_expect("split");
    DecimalByteParts::try_new_with_lower_parts(
        parts.msp,
        parts.lower_parts,
        decimal.decimal_dtype(),
    )
    .vortex_expect("valid byte parts")
}

/// A validity mask of exactly `len` entries, so null rows exercise lower parts holding bits
/// that must never be read.
fn draw_validity(tc: &TestCase, len: usize) -> Validity {
    let valid: Vec<bool> = tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len));
    Validity::from_iter(valid)
}

/// An `i128`-backed decimal. The bounds keep values inside `Decimal(38, 2)` while still
/// reaching both sides of the 64-bit word boundary the encoding splits on.
fn draw_i128_decimal(tc: &TestCase) -> DecimalArray {
    let values: Vec<i128> = tc.draw(
        gs::vecs(
            gs::integers::<i128>()
                .min_value(-MAX_I128)
                .max_value(MAX_I128),
        )
        .min_size(1)
        .max_size(MAX_LEN),
    );
    let validity = draw_validity(tc, values.len());
    DecimalArray::new(Buffer::from(values), DecimalDType::new(38, 2), validity)
}

/// An `i256`-backed decimal, built from a signed high half and an unsigned low half so the
/// draw covers sign extension above the most significant part.
fn draw_i256_decimal(tc: &TestCase) -> DecimalArray {
    let halves: Vec<(i128, u128)> = tc.draw(
        gs::vecs(gs::tuples2(
            gs::integers::<i128>()
                .min_value(-MAX_I256_HIGH)
                .max_value(MAX_I256_HIGH),
            gs::integers::<u128>(),
        ))
        .min_size(1)
        .max_size(MAX_LEN),
    );
    let values: Vec<i256> = halves
        .into_iter()
        .map(|(high, low)| i256::from_parts(low, high))
        .collect();
    let validity = draw_validity(tc, values.len());
    DecimalArray::new(Buffer::from(values), DecimalDType::new(76, 2), validity)
}

fn draw_decimal(tc: &TestCase) -> DecimalArray {
    if tc.draw(gs::booleans()) {
        draw_i128_decimal(tc)
    } else {
        draw_i256_decimal(tc)
    }
}

/// Canonicalize an encoded array back to a `DecimalArray`.
fn canonicalize(array: ArrayRef, ctx: &mut ExecutionCtx) -> DecimalArray {
    array.execute::<DecimalArray>(ctx).vortex_expect("execute")
}

/// A byte-parts array built directly from drawn parts, rather than by splitting a decimal.
///
/// `split_decimal` only ever emits 0, 1 or 3 lower parts under an `i64` most significant
/// part, so drawing the part count here is the only way to reach the two-part shape and the
/// sign extension that sits above a most significant part below the top word.
fn draw_encoded(tc: &TestCase) -> (DecimalBytePartsArray, usize) {
    let lower_part_count = tc.draw(gs::integers::<usize>().min_value(0).max_value(3));
    let msp: Vec<i64> = tc.draw(
        gs::vecs(gs::integers::<i64>())
            .min_size(1)
            .max_size(MAX_LEN),
    );
    let len = msp.len();

    let lower: Vec<ArrayRef> = (0..lower_part_count)
        .map(|_| {
            let part: Vec<u64> =
                tc.draw(gs::vecs(gs::integers::<u64>()).min_size(len).max_size(len));
            PrimitiveArray::new(Buffer::from(part), Validity::NonNullable).into_array()
        })
        .collect();

    // The declared precision must be wide enough for what the parts assemble into.
    let precision = match lower_part_count {
        0 => 18,
        1 => 38,
        _ => 76,
    };
    let msp = PrimitiveArray::new(Buffer::from(msp), draw_validity(tc, len)).into_array();
    let array =
        DecimalByteParts::try_new_with_lower_parts(msp, lower, DecimalDType::new(precision, 2))
            .vortex_expect("valid byte parts");
    (array, len)
}

/// Encoding a decimal and decoding it again must reproduce it exactly, including null rows
/// and the storage width.
#[hegel::test]
fn decoded_survives_encode_then_decode(tc: TestCase) {
    let decimal = draw_decimal(&tc);
    let mut ctx = ctx();

    let round_tripped = canonicalize(encode(&decimal).into_array(), &mut ctx);

    assert_eq!(round_tripped.values_type(), decimal.values_type());
    assert_arrays_eq!(decimal, round_tripped, &mut ctx);
}

/// Decoding an encoded array and encoding it again must not change the values it decodes to.
///
/// Starting from the encoded side reaches part counts `split_decimal` never produces, so this
/// covers layouts the property above cannot generate. It compares decoded values rather than
/// the arrays themselves because re-encoding normalizes the part count: splitting an `i256`
/// always yields three lower parts, whatever the original array carried.
#[hegel::test]
fn encoded_survives_decode_then_encode(tc: TestCase) {
    let (array, _len) = draw_encoded(&tc);
    let mut ctx = ctx();

    let decoded = canonicalize(array.into_array(), &mut ctx);
    let re_decoded = canonicalize(encode(&decoded).into_array(), &mut ctx);

    assert_arrays_eq!(decoded, re_decoded, &mut ctx);
}

// TODO(joe): restore the coverage removed alongside these two round trips. Each of the
// following was a property here and caught mutations that the round trips do not:
//
// - `scalar_at` against bulk canonicalization. `combine_i128`/`combine_i256` are a second
//   implementation of the assembly loops and can drift from them silently.
// - filter, slice and take against the same operation on the canonical array. These caught
//   part-order and word-placement mutations, though the round trips catch those too.
// - a serialize/decode round trip, which is the only property that exercised the metadata
//   carrying the lower part count.
// - sign extension above a most significant part below the top word, checked against an
//   expectation computed independently of the assembly loop. This is the one real gap: a
//   round trip compares decode against decode, so a decode-side sign-extension bug is
//   invisible to it. Dropping the sign fill in `sign_extended_words` is caught by neither
//   property here.

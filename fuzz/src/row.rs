// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fuzzer for the row-oriented byte encoder (`vortex-row`).
//!
//! Two related workloads share a single input generator ([`FuzzRowEncode`]):
//!
//! - [`run_row_encode`] exercises [`RowEncoder::encode`] on arbitrary equal-length columns and
//!   checks basic invariants (row count and determinism). Its primary job is to surface panics
//!   and internal errors on the full range of supported logical types.
//! - [`run_row_encode_compress`] additionally BtrBlocks-compresses every column and checks that
//!   row-encoding the compressed columns produces byte-identical rows to row-encoding the
//!   originals. Compression is value-preserving and row keys are a pure function of the logical
//!   values, so the two encodings must match byte for byte.

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_row::RowEncoder;
use vortex_row::RowEncodingOptions;
use vortex_row::RowSortField;

use crate::FUZZ_ARRAY_MAX_LEN;
use crate::SESSION;
use crate::array::assert_array_eq;
use crate::error::Backtrace;
use crate::error::VortexFuzzError;
use crate::error::VortexFuzzResult;

/// Maximum number of input columns to row-encode in a single fuzz input.
const MAX_COLUMNS: usize = 3;

/// Input for the row-encoding fuzzers: a set of equal-length columns plus one
/// [`RowSortField`] per column.
#[derive(Clone, Debug)]
pub struct FuzzRowEncode {
    /// The columns to row-encode. All have the same length.
    pub columns: Vec<ArrayRef>,
    /// One sort field per column, in column order.
    pub options: RowEncodingOptions,
}

impl<'a> Arbitrary<'a> for FuzzRowEncode {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // The first column fixes the shared row count; the remaining columns reuse it so the
        // encoder receives equal-length inputs.
        let first = ArbitraryArray::arbitrary_with_config(
            u,
            &ArbitraryArrayConfig {
                dtype: None,
                len: 0..=FUZZ_ARRAY_MAX_LEN,
            },
        )?
        .0;
        let nrows = first.len();

        let ncols = u.int_in_range(1..=MAX_COLUMNS)?;
        let mut columns = Vec::with_capacity(ncols);
        columns.push(first);
        for _ in 1..ncols {
            let col = ArbitraryArray::arbitrary_with_config(
                u,
                &ArbitraryArrayConfig {
                    dtype: None,
                    len: nrows..=nrows,
                },
            )?
            .0;
            columns.push(col);
        }

        let fields = (0..ncols)
            .map(|_| Ok(RowSortField::new(u.arbitrary()?, u.arbitrary()?)))
            .collect::<arbitrary::Result<Vec<RowSortField>>>()?;

        Ok(FuzzRowEncode {
            columns,
            options: RowEncodingOptions::new(fields),
        })
    }
}

/// Attempt to row-encode `columns`. A returned `Ok(None)` means the encoder rejected the input
/// for an expected reason (an unsupported logical type such as extension/variant/union/variable
/// list, or an input that exceeds the `u32` size limits). Those are uninteresting inputs rather
/// than bugs, so callers reject them from the corpus.
#[expect(clippy::result_large_err)]
fn try_encode(
    encoder: &RowEncoder,
    columns: &[ArrayRef],
    ctx: &mut ExecutionCtx,
) -> VortexFuzzResult<Option<ListViewArray>> {
    match encoder.encode(columns, ctx) {
        Ok(encoded) => Ok(Some(encoded)),
        // Unsupported dtype or out-of-range input: expected, not a bug.
        Err(_) => Ok(None),
    }
}

/// Run the row-encoding fuzzer.
///
/// Returns:
/// - `Ok(true)` - keep in corpus
/// - `Ok(false)` - reject from corpus (e.g. an unsupported logical type)
/// - `Err(_)` - a bug was found
#[expect(clippy::result_large_err)]
pub fn run_row_encode(fuzz: FuzzRowEncode) -> VortexFuzzResult<bool> {
    let FuzzRowEncode { columns, options } = fuzz;
    let mut ctx = SESSION.create_execution_ctx();
    let encoder = RowEncoder::with_options(options);

    let Some(encoded) = try_encode(&encoder, &columns, &mut ctx)? else {
        return Ok(false);
    };

    // The encoded ListView holds exactly one row per input row.
    let nrows = columns[0].len();
    let encoded = encoded.into_array();
    if encoded.len() != nrows {
        return Err(VortexFuzzError::LengthMismatch(
            nrows,
            encoded.len(),
            columns[0].clone(),
            encoded,
            0,
            Backtrace::capture(),
        ));
    }

    // Encoding is a pure function of the inputs and options, so a second pass must produce
    // byte-identical rows.
    let encoded_again = encoder
        .encode(&columns, &mut ctx)
        .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?
        .into_array();
    assert_array_eq(&encoded, &encoded_again, 0)?;

    Ok(true)
}

/// Run the compression-differential row-encoding fuzzer.
///
/// Row-encodes the original columns and the BtrBlocks-compressed columns and asserts that the
/// two encodings are byte-identical. Compression preserves logical values, so the row keys must
/// match.
///
/// Returns:
/// - `Ok(true)` - keep in corpus
/// - `Ok(false)` - reject from corpus (e.g. an unsupported logical type)
/// - `Err(_)` - a bug was found
#[expect(clippy::result_large_err)]
pub fn run_row_encode_compress(fuzz: FuzzRowEncode) -> VortexFuzzResult<bool> {
    let FuzzRowEncode { columns, options } = fuzz;
    let mut ctx = SESSION.create_execution_ctx();
    let encoder = RowEncoder::with_options(options);

    // Baseline: row-encode the original columns. Reject inputs the encoder does not support.
    let Some(original) = try_encode(&encoder, &columns, &mut ctx)? else {
        return Ok(false);
    };

    // Compress each column with BtrBlocks. Canonicalize first to match the array_ops fuzzer's
    // Compress action and to give the compressor a stable starting point.
    let mut compressed_columns = Vec::with_capacity(columns.len());
    for col in &columns {
        let canonical = col
            .clone()
            .execute::<Canonical>(&mut ctx)
            .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?
            .into_array();
        let compressed = BtrBlocksCompressor::default()
            .compress(&canonical, &mut ctx)
            .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?;
        compressed_columns.push(compressed);
    }

    // The original encode succeeded, so the (dtype-preserving) compressed columns must encode
    // too; an error here is a real bug rather than an unsupported input.
    let compressed_encoded = encoder
        .encode(&compressed_columns, &mut ctx)
        .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?
        .into_array();

    assert_array_eq(&original.into_array(), &compressed_encoded, 0)?;

    Ok(true)
}

#[cfg(test)]
mod tests {
    //! A lightweight in-process driver that exercises the row-encoding fuzz workloads over many
    //! pseudo-random inputs without the `cargo-fuzz` harness. It mirrors what `cargo fuzz run
    //! row_encode` / `row_encode_compress` would do, so it can surface panics and oracle
    //! failures in CI and locally.

    use arbitrary::Arbitrary;
    use arbitrary::Unstructured;

    use super::FuzzRowEncode;
    use super::run_row_encode;
    use super::run_row_encode_compress;

    /// Deterministic xorshift64* generator: enough entropy to drive `Arbitrary` without pulling
    /// in an RNG dependency.
    fn next(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn random_bytes(state: &mut u64, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            out.extend_from_slice(&next(state).to_le_bytes());
        }
        out.truncate(len);
        out
    }

    /// Print a column summary for a failing input to make CI failures actionable.
    fn report_failure(label: &str, i: usize, fuzz: &FuzzRowEncode, err: &dyn std::fmt::Display) {
        eprintln!("=== {label} failed on iteration {i} ===");
        for (c, col) in fuzz.columns.iter().enumerate() {
            eprintln!(
                "col {c} dtype={} field={:?}",
                col.dtype(),
                fuzz.options.fields()[c]
            );
            eprintln!("{}", col.display_tree());
        }
        panic!("{label} failed on iteration {i}: {err}");
    }

    #[test]
    fn driver_row_encode_and_compress() {
        const ITERATIONS: usize = 400;
        // A couple of independent seed streams broaden the explored input space.
        for seed in [0x9E37_79B9_7F4A_7C15u64, 0xD1B5_4A32_D192_ED03] {
            let mut state = seed;
            for i in 0..ITERATIONS {
                // Vary the input size so the generator explores different array shapes.
                let len = 32 + (next(&mut state) as usize % 4096);
                let data = random_bytes(&mut state, len);
                let mut u = Unstructured::new(&data);

                let Ok(fuzz) = FuzzRowEncode::arbitrary(&mut u) else {
                    continue;
                };

                if let Err(e) = run_row_encode(fuzz.clone()) {
                    report_failure("run_row_encode", i, &fuzz, &e);
                }
                if let Err(e) = run_row_encode_compress(fuzz.clone()) {
                    report_failure("run_row_encode_compress", i, &fuzz, &e);
                }
            }
        }
    }
}

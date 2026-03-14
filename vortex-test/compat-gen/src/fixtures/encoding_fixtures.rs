// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-encoding synthetic fixtures.
//!
//! Each fixture produces data patterns designed to exercise a specific stable encoding.
//! The `expected_encodings` method declares which encoding(s) the Vortex compressor
//! should select for this data.

use std::path::Path;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::FieldNames;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::validity::Validity;
use vortex_array::vtable::ArrayId;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_layout::LayoutId;

use super::ExpectedEncoding;
use super::Fixture;

const N: usize = 1024;

/// All per-encoding fixtures.
pub fn all_encoding_fixtures() -> Vec<Box<dyn Fixture>> {
    vec![
        Box::new(AlpFixture),
        Box::new(AlprdFixture),
        Box::new(BitPackedFixture),
        Box::new(ByteBoolFixture),
        Box::new(DateTimePartsFixture),
        Box::new(DecimalBytePartsFixture),
        Box::new(DeltaFixture),
        Box::new(DictFixture),
        Box::new(FsstFixture),
        Box::new(FoRFixture),
        Box::new(PcoFixture),
        Box::new(RleFixture),
        Box::new(RunEndFixture),
        Box::new(SequenceFixture),
        Box::new(SparseFixture),
        Box::new(ZigZagFixture),
        Box::new(ConstantFixture),
        // Layout-oriented fixtures
        Box::new(FlatLayoutFixture),
        Box::new(ChunkedLayoutFixture),
        Box::new(DictLayoutFixture),
        Box::new(StructLayoutFixture),
    ]
}

// ---------------------------------------------------------------------------
// ALP: Adaptive Lossless floating-Point compression
// ---------------------------------------------------------------------------

pub struct AlpFixture;

impl Fixture for AlpFixture {
    fn name(&self) -> &str {
        "enc_alp.vortex"
    }

    fn description(&self) -> &str {
        "Near-integer floats and decimal-like prices for ALP encoding (f32 + f64)"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref("vortex.alp"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let f64_prices: Vec<f64> = (0..N).map(|i| 100.0 + (i as f64) * 0.25).collect();
        let f32_near_int: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let f64_currency: Vec<f64> = (0..N).map(|i| ((i % 10000) as f64) / 100.0).collect();

        let arr = StructArray::try_new(
            FieldNames::from(["f64_prices", "f32_near_int", "f64_currency"]),
            vec![
                PrimitiveArray::new(Buffer::from(f64_prices), Validity::NonNullable).into_array(),
                PrimitiveArray::new(Buffer::from(f32_near_int), Validity::NonNullable).into_array(),
                PrimitiveArray::new(Buffer::from(f64_currency), Validity::NonNullable).into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// ALPRD: ALP with Real Doubles (reduced precision delta)
// ---------------------------------------------------------------------------

pub struct AlprdFixture;

impl Fixture for AlprdFixture {
    fn name(&self) -> &str {
        "enc_alprd.vortex"
    }

    fn description(&self) -> &str {
        "Real-valued doubles with small deltas for ALPRD encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref("vortex.alprd"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let sensor: Vec<f64> = (0..N)
            .map(|i| {
                let noise = ((i * 7 + 13) % 100) as f64 / 1000.0;
                98.6 + noise
            })
            .collect();

        let drift: Vec<f64> = (0..N)
            .map(|i| 1000.0 + (i as f64) * 0.001 + ((i * 3) % 7) as f64 * 0.0001)
            .collect();

        let arr = StructArray::try_new(
            FieldNames::from(["sensor", "drift"]),
            vec![
                PrimitiveArray::new(Buffer::from(sensor), Validity::NonNullable).into_array(),
                PrimitiveArray::new(Buffer::from(drift), Validity::NonNullable).into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// BitPacked: Fastlanes bit-packing for unsigned integers
// ---------------------------------------------------------------------------

pub struct BitPackedFixture;

impl Fixture for BitPackedFixture {
    fn name(&self) -> &str {
        "enc_bitpacked.vortex"
    }

    fn description(&self) -> &str {
        "Small unsigned integers that fit in fewer bits than their type width"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref(
            "fastlanes.bitpacked",
        ))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let u32_8bit: Vec<u32> = (0..N as u32).map(|i| i % 256).collect();
        let u64_12bit: Vec<u64> = (0..N as u64).map(|i| i % 4096).collect();
        let u16_4bit: Vec<u16> = (0..N as u16).map(|i| i % 16).collect();

        let arr = StructArray::try_new(
            FieldNames::from(["u32_8bit", "u64_12bit", "u16_4bit"]),
            vec![
                PrimitiveArray::new(Buffer::from(u32_8bit), Validity::NonNullable).into_array(),
                PrimitiveArray::new(Buffer::from(u64_12bit), Validity::NonNullable).into_array(),
                PrimitiveArray::new(Buffer::from(u16_4bit), Validity::NonNullable).into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// ByteBool: byte-per-boolean encoding
// ---------------------------------------------------------------------------

pub struct ByteBoolFixture;

impl Fixture for ByteBoolFixture {
    fn name(&self) -> &str {
        "enc_bytebool.vortex"
    }

    fn description(&self) -> &str {
        "Boolean arrays for ByteBool encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref("vortex.bytebool"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let alternating = BoolArray::from_iter((0..N).map(|i| i % 2 == 0));
        let mostly_true = BoolArray::from_iter((0..N).map(|i| i % 100 != 0));
        let mixed = BoolArray::from_iter((0..N).map(|i| (i * 7 + 3) % 5 > 1));

        let arr = StructArray::try_new(
            FieldNames::from(["alternating", "mostly_true", "mixed"]),
            vec![
                alternating.into_array(),
                mostly_true.into_array(),
                mixed.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// DateTimeParts: temporal decomposition encoding
// ---------------------------------------------------------------------------

pub struct DateTimePartsFixture;

impl Fixture for DateTimePartsFixture {
    fn name(&self) -> &str {
        "enc_datetimeparts.vortex"
    }

    fn description(&self) -> &str {
        "Timestamp arrays (microsecond and nanosecond) for DateTimeParts encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref(
            "vortex.datetimeparts",
        ))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let base_us: i64 = 1_704_067_200_000_000; // 2024-01-01T00:00:00 in microseconds
        let ts_us: Vec<i64> = (0..N as i64).map(|i| base_us + i * 3_600_000_000).collect();
        let ts_us_arr = TemporalArray::new_timestamp(
            PrimitiveArray::new(Buffer::from(ts_us), Validity::NonNullable).into_array(),
            TimeUnit::Microseconds,
            None,
        );

        let base_ns: i64 = 1_704_067_200_000_000_000; // 2024-01-01T00:00:00 in nanoseconds
        let ts_ns: Vec<i64> = (0..N as i64).map(|i| base_ns + i * 1_000_000_000).collect();
        let ts_ns_arr = TemporalArray::new_timestamp(
            PrimitiveArray::new(Buffer::from(ts_ns), Validity::NonNullable).into_array(),
            TimeUnit::Nanoseconds,
            None,
        );

        let arr = StructArray::try_new(
            FieldNames::from(["ts_us", "ts_ns"]),
            vec![ts_us_arr.into_array(), ts_ns_arr.into_array()],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// DecimalByteParts: decimal decomposition encoding
// ---------------------------------------------------------------------------

pub struct DecimalBytePartsFixture;

impl Fixture for DecimalBytePartsFixture {
    fn name(&self) -> &str {
        "enc_decimal_byte_parts.vortex"
    }

    fn description(&self) -> &str {
        "Fixed-precision decimal arrays for DecimalByteParts encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref(
            "vortex.decimal_byte_parts",
        ))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let values: Vec<i64> = (0..N as i64).map(|i| i * 100 + (i % 100)).collect();
        let decimal_arr = DecimalArray::from_iter(values, decimal_dtype);

        let hi_prec_dtype = DecimalDType::new(18, 6);
        let hi_prec_values: Vec<i64> = (0..N as i64)
            .map(|i| i * 1_000_000 + (i * 7 % 999_999))
            .collect();
        let hi_prec_arr = DecimalArray::from_iter(hi_prec_values, hi_prec_dtype);

        let arr = StructArray::try_new(
            FieldNames::from(["dec_10_2", "dec_18_6"]),
            vec![decimal_arr.into_array(), hi_prec_arr.into_array()],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// Delta: Fastlanes delta encoding for sorted/monotonic integers
// ---------------------------------------------------------------------------

pub struct DeltaFixture;

impl Fixture for DeltaFixture {
    fn name(&self) -> &str {
        "enc_delta.vortex"
    }

    fn description(&self) -> &str {
        "Monotonically increasing and sorted integers for Delta encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref("fastlanes.delta"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let monotonic_u64: Vec<u64> = (0..N as u64).map(|i| i * 3 + 1000).collect();
        let sorted_i32: Vec<i32> = (0..N as i32).map(|i| -500 + i + (i / 100)).collect();
        let sorted_i64: Vec<i64> = (0..N as i64).map(|i| 1_700_000_000 + i * 60).collect();

        let arr = StructArray::try_new(
            FieldNames::from(["monotonic_u64", "sorted_i32", "sorted_i64"]),
            vec![
                PrimitiveArray::new(Buffer::from(monotonic_u64), Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::new(Buffer::from(sorted_i32), Validity::NonNullable).into_array(),
                PrimitiveArray::new(Buffer::from(sorted_i64), Validity::NonNullable).into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// Dict: dictionary/categorical encoding
// ---------------------------------------------------------------------------

pub struct DictFixture;

impl Fixture for DictFixture {
    fn name(&self) -> &str {
        "enc_dict.vortex"
    }

    fn description(&self) -> &str {
        "Low-cardinality repeated values (strings and integers) for Dict encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref("vortex.dict"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let categories = ["red", "green", "blue", "yellow", "purple"];
        let str_values: Vec<&str> = (0..N).map(|i| categories[i % categories.len()]).collect();
        let str_col = VarBinArray::from(str_values);

        let int_values: Vec<i32> = (0..N as i32).map(|i| (i % 10) * 100).collect();
        let int_col = PrimitiveArray::new(Buffer::from(int_values), Validity::NonNullable);

        let nullable_values: Vec<Option<&str>> = (0..N)
            .map(|i| (i % 7 != 0).then_some(categories[i % categories.len()]))
            .collect();
        let nullable_col = VarBinArray::from(nullable_values);

        let arr = StructArray::try_new(
            FieldNames::from(["str_cat", "int_cat", "nullable_cat"]),
            vec![
                str_col.into_array(),
                int_col.into_array(),
                nullable_col.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// FSST: Fast Static Symbol Table string compression
// ---------------------------------------------------------------------------

pub struct FsstFixture;

impl Fixture for FsstFixture {
    fn name(&self) -> &str {
        "enc_fsst.vortex"
    }

    fn description(&self) -> &str {
        "Strings with common substrings/prefixes for FSST encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref("vortex.fsst"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let prefixes = [
            "https://example.com/api/v1/users/",
            "https://example.com/api/v1/orders/",
            "https://example.com/api/v1/products/",
            "https://example.com/api/v2/users/",
        ];
        let urls: Vec<String> = (0..N)
            .map(|i| format!("{}{}", prefixes[i % prefixes.len()], i))
            .collect();
        let url_refs: Vec<&str> = urls.iter().map(|s| s.as_str()).collect();
        let url_col = VarBinArray::from(url_refs);

        let severities = ["INFO", "WARN", "ERROR", "DEBUG"];
        let components = ["auth", "db", "cache", "api"];
        let logs: Vec<String> = (0..N)
            .map(|i| {
                format!(
                    "[{}] {}: request processed in {}ms",
                    severities[i % severities.len()],
                    components[i % components.len()],
                    i % 1000
                )
            })
            .collect();
        let log_refs: Vec<&str> = logs.iter().map(|s| s.as_str()).collect();
        let log_col = VarBinArray::from(log_refs);

        let arr = StructArray::try_new(
            FieldNames::from(["urls", "logs"]),
            vec![url_col.into_array(), log_col.into_array()],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// FoR: Frame-of-Reference encoding
// ---------------------------------------------------------------------------

pub struct FoRFixture;

impl Fixture for FoRFixture {
    fn name(&self) -> &str {
        "enc_for.vortex"
    }

    fn description(&self) -> &str {
        "Integers clustered around a base value for Frame-of-Reference encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref("fastlanes.for"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let clustered_i32: Vec<i32> = (0..N as i32).map(|i| 1_000_000 + (i % 100)).collect();
        let clustered_u64: Vec<u64> = (0..N as u64).map(|i| 10_000_000_000 + (i % 256)).collect();
        let clustered_i64: Vec<i64> = (0..N as i64).map(|i| 1_704_067_200 + (i % 3600)).collect();

        let arr = StructArray::try_new(
            FieldNames::from(["clustered_i32", "clustered_u64", "clustered_i64"]),
            vec![
                PrimitiveArray::new(Buffer::from(clustered_i32), Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::new(Buffer::from(clustered_u64), Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::new(Buffer::from(clustered_i64), Validity::NonNullable)
                    .into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// Pco: Patas Compression Optimizer
// ---------------------------------------------------------------------------

pub struct PcoFixture;

impl Fixture for PcoFixture {
    fn name(&self) -> &str {
        "enc_pco.vortex"
    }

    fn description(&self) -> &str {
        "Various numeric patterns for Pco (patas compression) encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref("vortex.pco"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let irregular_i64: Vec<i64> = (0..N as i64).map(|i| i * i + (i % 17) * 1000).collect();
        let smooth_f64: Vec<f64> = (0..N)
            .map(|i| {
                let t = i as f64 / N as f64;
                t * t * (3.0 - 2.0 * t) * 100.0
            })
            .collect();
        let pattern_u32: Vec<u32> = (0..N as u32)
            .map(|i| i.wrapping_mul(2_654_435_761) % 65536)
            .collect();

        let arr = StructArray::try_new(
            FieldNames::from(["irregular_i64", "smooth_f64", "pattern_u32"]),
            vec![
                PrimitiveArray::new(Buffer::from(irregular_i64), Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::new(Buffer::from(smooth_f64), Validity::NonNullable).into_array(),
                PrimitiveArray::new(Buffer::from(pattern_u32), Validity::NonNullable).into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// RLE: Fastlanes Run-Length Encoding
// ---------------------------------------------------------------------------

pub struct RleFixture;

impl Fixture for RleFixture {
    fn name(&self) -> &str {
        "enc_rle.vortex"
    }

    fn description(&self) -> &str {
        "Data with long runs of repeated values for RLE encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref("fastlanes.rle"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let runs_i32: Vec<i32> = (0..N as i32).map(|i| i / 64).collect();
        let labels = ["active", "inactive", "pending"];
        let runs_str: Vec<&str> = (0..N).map(|i| labels[i / 341 % labels.len()]).collect();
        let str_col = VarBinArray::from(runs_str);
        let runs_bool = BoolArray::from_iter((0..N).map(|i| (i / 128) % 2 == 0));

        let arr = StructArray::try_new(
            FieldNames::from(["runs_i32", "runs_str", "runs_bool"]),
            vec![
                PrimitiveArray::new(Buffer::from(runs_i32), Validity::NonNullable).into_array(),
                str_col.into_array(),
                runs_bool.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// RunEnd: run-end encoding
// ---------------------------------------------------------------------------

pub struct RunEndFixture;

impl Fixture for RunEndFixture {
    fn name(&self) -> &str {
        "enc_runend.vortex"
    }

    fn description(&self) -> &str {
        "Data with variable-length runs for RunEnd encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref("vortex.runend"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let run_lengths = [1usize, 5, 10, 50, 100];
        let mut values = Vec::with_capacity(N);
        let mut run_idx = 0i64;
        let mut rl_idx = 0;
        while values.len() < N {
            let run_len = run_lengths[rl_idx % run_lengths.len()].min(N - values.len());
            for _ in 0..run_len {
                values.push(run_idx);
            }
            run_idx += 1;
            rl_idx += 1;
        }
        let run_col = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);

        let statuses = ["open", "closed", "pending", "cancelled"];
        let mut status_values = Vec::with_capacity(N);
        let mut s_idx = 0;
        let mut remaining = N;
        while remaining > 0 {
            let run_len = (32 + s_idx * 7 % 64).min(remaining);
            for _ in 0..run_len {
                status_values.push(statuses[s_idx % statuses.len()]);
            }
            s_idx += 1;
            remaining -= run_len;
        }
        let status_col = VarBinArray::from(status_values);

        let arr = StructArray::try_new(
            FieldNames::from(["run_values", "statuses"]),
            vec![run_col.into_array(), status_col.into_array()],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// Sequence: arithmetic sequence encoding
// ---------------------------------------------------------------------------

pub struct SequenceFixture;

impl Fixture for SequenceFixture {
    fn name(&self) -> &str {
        "enc_sequence.vortex"
    }

    fn description(&self) -> &str {
        "Arithmetic sequences (0,1,2,... and stepped) for Sequence encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref("vortex.sequence"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let row_ids: Vec<u64> = (0..N as u64).collect();
        let stepped: Vec<i32> = (0..N as i32).map(|i| i * 5).collect();
        let offset: Vec<i64> = (0..N as i64).map(|i| i + 1000).collect();

        let arr = StructArray::try_new(
            FieldNames::from(["row_ids", "stepped", "offset"]),
            vec![
                PrimitiveArray::new(Buffer::from(row_ids), Validity::NonNullable).into_array(),
                PrimitiveArray::new(Buffer::from(stepped), Validity::NonNullable).into_array(),
                PrimitiveArray::new(Buffer::from(offset), Validity::NonNullable).into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// Sparse: sparse encoding for mostly-default arrays
// ---------------------------------------------------------------------------

pub struct SparseFixture;

impl Fixture for SparseFixture {
    fn name(&self) -> &str {
        "enc_sparse.vortex"
    }

    fn description(&self) -> &str {
        "Mostly-null or mostly-default arrays with sparse non-default values"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref("vortex.sparse"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let sparse_i64_col = PrimitiveArray::from_option_iter(
            (0..N as i64).map(|i| (i % 50 == 0).then_some(i * 1000)),
        );

        let sparse_str: Vec<Option<&str>> = (0..N)
            .map(|i| (i % 20 == 0).then_some("rare_value"))
            .collect();
        let sparse_str_col = VarBinArray::from(sparse_str);

        let sparse_bool_col = BoolArray::from_iter((0..N).map(|i| (i % 100 == 0).then_some(true)));

        let arr = StructArray::try_new(
            FieldNames::from(["sparse_i64", "sparse_str", "sparse_bool"]),
            vec![
                sparse_i64_col.into_array(),
                sparse_str_col.into_array(),
                sparse_bool_col.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// ZigZag: signed-to-unsigned encoding for small absolute values
// ---------------------------------------------------------------------------

pub struct ZigZagFixture;

impl Fixture for ZigZagFixture {
    fn name(&self) -> &str {
        "enc_zigzag.vortex"
    }

    fn description(&self) -> &str {
        "Signed integers with small absolute values for ZigZag encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref("vortex.zigzag"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let alternating_i32: Vec<i32> = (0..N as i32)
            .map(|i| {
                let v = i / 2 + 1;
                if i % 2 == 0 { v } else { -v }
            })
            .collect();
        let small_i64: Vec<i64> = (0..N as i64).map(|i| (i % 21) - 10).collect();
        let deltas_i32: Vec<i32> = (0..N as i32).map(|i| -(i % 50)).collect();

        let arr = StructArray::try_new(
            FieldNames::from(["alternating_i32", "small_i64", "deltas_i32"]),
            vec![
                PrimitiveArray::new(Buffer::from(alternating_i32), Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::new(Buffer::from(small_i64), Validity::NonNullable).into_array(),
                PrimitiveArray::new(Buffer::from(deltas_i32), Validity::NonNullable).into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ---------------------------------------------------------------------------
// Constant: constant-value arrays (used as a btrblocks compression scheme)
// ---------------------------------------------------------------------------

pub struct ConstantFixture;

impl Fixture for ConstantFixture {
    fn name(&self) -> &str {
        "enc_constant.vortex"
    }

    fn description(&self) -> &str {
        "Constant-value columns (int, float, string, bool, null) for Constant encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Array(ArrayId::new_ref("vortex.constant"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let const_i32 = ConstantArray::new(42i32, N);
        let const_f64 = ConstantArray::new(99.99f64, N);
        let const_bool = ConstantArray::new(true, N);
        let const_str = ConstantArray::new("constant_value", N);

        let arr = StructArray::try_new(
            FieldNames::from(["const_i32", "const_f64", "const_bool", "const_str"]),
            vec![
                const_i32.into_array(),
                const_f64.into_array(),
                const_bool.into_array(),
                const_str.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

// ===========================================================================
// Layout-oriented fixtures
// ===========================================================================

pub struct FlatLayoutFixture;

impl Fixture for FlatLayoutFixture {
    fn name(&self) -> &str {
        "layout_flat.vortex"
    }

    fn description(&self) -> &str {
        "Single small array that fits in one flat layout segment"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Layout(LayoutId::new_ref("vortex.flat"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let values: Vec<i32> = (0..64).collect();
        let arr = StructArray::try_new(
            FieldNames::from(["value"]),
            vec![PrimitiveArray::new(Buffer::from(values), Validity::NonNullable).into_array()],
            64,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

pub struct ChunkedLayoutFixture;

impl Fixture for ChunkedLayoutFixture {
    fn name(&self) -> &str {
        "layout_chunked.vortex"
    }

    fn description(&self) -> &str {
        "Multiple chunks of mixed types exercising chunked layout"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Layout(LayoutId::new_ref(
            "vortex.chunked",
        ))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        (0i64..5)
            .map(|chunk_idx| {
                let base = chunk_idx * 512;
                let ints: Vec<i64> = (0i64..512).map(|i| base + i).collect();
                let strs: Vec<String> = (0..512)
                    .map(|i| format!("chunk{}_{}", chunk_idx, i))
                    .collect();
                let str_refs: Vec<&str> = strs.iter().map(|s| s.as_str()).collect();
                let bools = BoolArray::from_iter(
                    (0..512usize).map(|i| (i + chunk_idx as usize).is_multiple_of(3)),
                );

                Ok(StructArray::try_new(
                    FieldNames::from(["id", "label", "flag"]),
                    vec![
                        PrimitiveArray::new(Buffer::from(ints), Validity::NonNullable).into_array(),
                        VarBinArray::from(str_refs).into_array(),
                        bools.into_array(),
                    ],
                    512,
                    Validity::NonNullable,
                )?
                .into_array())
            })
            .collect()
    }
}

pub struct DictLayoutFixture;

impl Fixture for DictLayoutFixture {
    fn name(&self) -> &str {
        "layout_dict.vortex"
    }

    fn description(&self) -> &str {
        "Very low cardinality strings to trigger dict layout encoding"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![
            ExpectedEncoding::Layout(LayoutId::new_ref("vortex.dict")),
            ExpectedEncoding::Array(ArrayId::new_ref("vortex.dict")),
        ]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let n = 2048;
        let values = ["alpha", "beta", "gamma"];
        let str_values: Vec<&str> = (0..n).map(|i| values[i % values.len()]).collect();
        let str_col = VarBinArray::from(str_values);
        let int_values: Vec<i32> = (0..n as i32).collect();

        let arr = StructArray::try_new(
            FieldNames::from(["category", "idx"]),
            vec![
                str_col.into_array(),
                PrimitiveArray::new(Buffer::from(int_values), Validity::NonNullable).into_array(),
            ],
            n,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

pub struct StructLayoutFixture;

impl Fixture for StructLayoutFixture {
    fn name(&self) -> &str {
        "layout_struct.vortex"
    }

    fn description(&self) -> &str {
        "Deeply nested structs to exercise struct layout decomposition"
    }

    fn expected_encodings(&self) -> Vec<ExpectedEncoding> {
        vec![ExpectedEncoding::Layout(LayoutId::new_ref("vortex.struct"))]
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let n = 256;
        let level2 = StructArray::try_new(
            FieldNames::from(["x", "y"]),
            vec![
                PrimitiveArray::new(
                    Buffer::from((0i32..n as i32).collect::<Vec<_>>()),
                    Validity::NonNullable,
                )
                .into_array(),
                PrimitiveArray::new(
                    Buffer::from((0i32..n as i32).map(|i| i * 2).collect::<Vec<_>>()),
                    Validity::NonNullable,
                )
                .into_array(),
            ],
            n,
            Validity::NonNullable,
        )?;

        let labels: Vec<String> = (0..n).map(|i| format!("item_{i}")).collect();
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        let level1 = StructArray::try_new(
            FieldNames::from(["coord", "label"]),
            vec![
                level2.into_array(),
                VarBinArray::from(label_refs).into_array(),
            ],
            n,
            Validity::NonNullable,
        )?;

        let arr = StructArray::try_new(
            FieldNames::from(["nested", "id"]),
            vec![
                level1.into_array(),
                PrimitiveArray::new(
                    Buffer::from((0u64..n as u64).collect::<Vec<_>>()),
                    Validity::NonNullable,
                )
                .into_array(),
            ],
            n,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

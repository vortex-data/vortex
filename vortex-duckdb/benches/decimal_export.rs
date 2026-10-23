// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Export of one decimal column into DuckDB data chunks.
//!
//! DuckDB stores a decimal in the width that its precision needs, with `i16` as the narrowest
//! width. A canonical array whose width matches DuckDB exports without a copy, and any other
//! array copies its values into the DuckDB vector. Each iteration builds the exporter, which
//! decodes the column, and then exports until the exporter is exhausted.

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::dtype::DecimalDType;
use vortex::dtype::NativeDecimalType;
use vortex::dtype::NativePType;
use vortex::encodings::decimal_byte_parts::DecimalByteParts;
use vortex::error::VortexExpect;
use vortex::session::VortexSession;
use vortex_duckdb::_benchmarking::ArrayExporter;
use vortex_duckdb::_benchmarking::ConversionCache;
use vortex_duckdb::duckdb::DataChunk;
use vortex_duckdb::duckdb::LogicalType;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

const ROWS: usize = 100_000;

#[derive(Clone, Copy)]
enum Case {
    /// Exports without a copy.
    CanonicalI64P18,
    /// Exports without a copy.
    CanonicalI16P4,
    /// Copies, because DuckDB stores precision 2 as `i16`.
    CanonicalI8P2,
    /// A DecimalByteParts array with an `i32` MSP under `Decimal(18, 2)`.
    MspI32P18,
    /// A DecimalByteParts array under `Decimal(10, 2)` whose values all fit an `i16` but need
    /// precision 5, above the precision 4 that `i16` holds.
    GapMspI16P10,
}

const CASES: &[Case] = &[
    Case::CanonicalI64P18,
    Case::CanonicalI16P4,
    Case::CanonicalI8P2,
    Case::MspI32P18,
    Case::GapMspI16P10,
];

impl Case {
    fn decimal_dtype(self) -> DecimalDType {
        match self {
            Self::CanonicalI64P18 | Self::MspI32P18 => DecimalDType::new(18, 2),
            Self::CanonicalI16P4 => DecimalDType::new(4, 2),
            Self::CanonicalI8P2 => DecimalDType::new(2, 1),
            Self::GapMspI16P10 => DecimalDType::new(10, 2),
        }
    }

    fn column(self) -> ArrayRef {
        let decimal_dtype = self.decimal_dtype();
        match self {
            Self::CanonicalI64P18 => {
                canonical(spread_values::<i64>(-MAX_P18, MAX_P18), decimal_dtype)
            }
            Self::CanonicalI16P4 => canonical(spread_values::<i16>(-9_999, 9_999), decimal_dtype),
            Self::CanonicalI8P2 => canonical(spread_values::<i8>(-99, 99), decimal_dtype),
            Self::MspI32P18 => decimal_byte_parts(
                spread_values::<i32>(-999_999_999, 999_999_999),
                decimal_dtype,
            ),
            Self::GapMspI16P10 => decimal_byte_parts(
                spread_values::<i16>(10_000, i64::from(i16::MAX)),
                decimal_dtype,
            ),
        }
    }
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::CanonicalI64P18 => "canonical_i64_p18",
            Self::CanonicalI16P4 => "canonical_i16_p4",
            Self::CanonicalI8P2 => "canonical_i8_p2",
            Self::MspI32P18 => "msp_i32_p18",
            Self::GapMspI16P10 => "gap_msp_i16_p10",
        })
    }
}

/// The largest magnitude of a `Decimal(18, _)` value.
const MAX_P18: i64 = 999_999_999_999_999_999;

/// `ROWS` values spread across `min..=max` in an unsorted, fixed order.
///
/// Row `i` holds `min + (i * STEP) % (max - min + 1)`. `STEP` is the prime `2^61 - 1`, so it is
/// coprime to every range size below it: consecutive rows land far apart, and the values repeat
/// only after every value of the range appears.
fn spread_values<T: TryFrom<i64>>(min: i64, max: i64) -> Buffer<T> {
    const STEP: i128 = (1 << 61) - 1;
    let count = i128::from(max) - i128::from(min) + 1;
    (0..ROWS)
        .map(|index| {
            let index = i128::try_from(index).vortex_expect("the row count fits an i128");
            let value = i64::try_from(i128::from(min) + (index * STEP) % count)
                .vortex_expect("every value lies in min..=max");
            T::try_from(value)
                .ok()
                .vortex_expect("every value fits the type of its case")
        })
        .collect()
}

fn canonical<T: NativeDecimalType>(values: Buffer<T>, decimal_dtype: DecimalDType) -> ArrayRef {
    DecimalArray::new(values, decimal_dtype, Validity::NonNullable).into_array()
}

fn decimal_byte_parts<T: NativePType>(msp: Buffer<T>, decimal_dtype: DecimalDType) -> ArrayRef {
    let msp = PrimitiveArray::new(msp, Validity::NonNullable).into_array();
    DecimalByteParts::try_new(msp, decimal_dtype)
        .vortex_expect("the MSP of every benchmark input is a signed integer array")
        .into_array()
}

#[divan::bench(args = CASES)]
fn decimal_export(bencher: Bencher, case: Case) {
    let decimal_dtype = case.decimal_dtype();
    let scale =
        u8::try_from(decimal_dtype.scale()).vortex_expect("every case has a non-negative scale");
    let logical_type = LogicalType::decimal_type(decimal_dtype.precision(), scale)
        .vortex_expect("DuckDB supports the precision of every case");
    let array = StructArray::from_fields(&[("value", case.column())])
        .vortex_expect("build the exported struct");
    let cache = ConversionCache::default();

    bencher
        .with_inputs(|| {
            (
                DataChunk::new([logical_type.clone()]),
                SESSION.create_execution_ctx(),
            )
        })
        .input_counter(|_| ItemsCount::new(ROWS))
        .bench_values(|(mut chunk, ctx)| {
            let mut exporter =
                ArrayExporter::try_new(&array, &cache, ctx).vortex_expect("build the exporter");
            while exporter
                .export(&mut chunk, None)
                .vortex_expect("export a chunk")
            {}
            // Returning the exporter keeps the drop of a copied column's decoded buffer out of the
            // timing, as it is for the columns that export without a copy.
            (exporter, chunk)
        });
}

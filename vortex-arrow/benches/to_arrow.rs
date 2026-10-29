// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::LazyLock;

use arrow_schema::DataType;
use arrow_schema::Field;
use divan::Bencher;
use divan::counter::ItemsCount;
use itertools::iproduct;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::WidenArray;
use vortex_array::builders::VarBinBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
#[expect(
    deprecated,
    reason = "benchmark comparing deprecated method with new one"
)]
use vortex_arrow::ArrowArrayExecutor;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::Buffer;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_onpair::DEFAULT_CONFIG;
use vortex_onpair::onpair_compress;
use vortex_session::VortexSession;
use vortex_zstd::Zstd;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>()
        .with::<ArrowSession>();
    vortex_decimal_byte_parts::initialize(&session);
    vortex_fsst::initialize(&session);
    vortex_onpair::initialize(&session);
    session.arrays().register(Zstd);
    session
});

fn schema() -> DType {
    let fields = StructFields::from_iter([
        (
            "primitive",
            DType::Primitive(PType::F32, Nullability::Nullable),
        ),
        (
            "list",
            DType::List(
                Arc::new(DType::Binary(Nullability::NonNullable)),
                Nullability::Nullable,
            ),
        ),
        (
            "decimal",
            DType::Decimal(DecimalDType::new(19, 10), Nullability::Nullable),
        ),
    ]);
    DType::Struct(fields, Nullability::NonNullable)
}

fn array() -> ArrayRef {
    StructArray::from_fields(&[
        (
            "primitive",
            PrimitiveArray::from_iter(0i16..1024).into_array(),
        ),
        (
            "list",
            ListArray::from_iter_slow::<u32, _>(
                (0..1024).map(|_| vec!["a", "b", "c"]).collect::<Vec<_>>(),
                Arc::new(DType::Utf8(Nullability::NonNullable)),
            )
            .unwrap()
            .into_array(),
        ),
        (
            "decimal",
            DecimalArray::from_iter(0i64..1024, DecimalDType::new(19, 2)).into_array(),
        ),
    ])
    .unwrap()
    .into_array()
}

#[allow(non_snake_case)]
#[divan::bench]
fn ArrowExportVTable_to_arrow_field(bencher: Bencher) {
    bencher
        .with_inputs(schema)
        .bench_values(|dtype| SESSION.arrow().to_arrow_field("", &dtype).unwrap())
}

#[derive(Clone, Copy)]
enum StringEncoding {
    Offset,
    View,
    Fsst,
    OnPair,
    Zstd,
}

impl Display for StringEncoding {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Offset => "offset",
            Self::View => "view",
            Self::Fsst => "fsst",
            Self::OnPair => "onpair",
            Self::Zstd => "zstd",
        })
    }
}

#[derive(Clone, Copy)]
enum StringStructure {
    Flat,
    Dict,
    Chunked,
}

impl Display for StringStructure {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Flat => "flat",
            Self::Dict => "dict",
            Self::Chunked => "chunked",
        })
    }
}

#[derive(Clone, Copy)]
enum StringValidity {
    NonNullable,
    Nullable,
}

impl StringValidity {
    fn nullability(self) -> Nullability {
        match self {
            Self::NonNullable => Nullability::NonNullable,
            Self::Nullable => Nullability::Nullable,
        }
    }
}

impl Display for StringValidity {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NonNullable => "nonnull",
            Self::Nullable => "nullable",
        })
    }
}

#[derive(Clone, Copy)]
struct StringCase {
    encoding: StringEncoding,
    structure: StringStructure,
    validity: StringValidity,
}

impl Display for StringCase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.encoding, self.structure, self.validity)
    }
}

#[derive(Clone, Copy)]
enum ArrowStringLayout {
    Offset,
    View,
}

impl ArrowStringLayout {
    fn data_type(self) -> DataType {
        match self {
            Self::Offset => DataType::Utf8,
            Self::View => DataType::Utf8View,
        }
    }
}

impl Display for ArrowStringLayout {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Offset => "offset",
            Self::View => "view",
        })
    }
}

#[derive(Clone, Copy)]
struct StringExportCase {
    array: StringCase,
    layout: ArrowStringLayout,
}

impl Display for StringExportCase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.layout, self.array)
    }
}

const STRING_ENCODINGS: &[StringEncoding] = &[
    StringEncoding::Offset,
    StringEncoding::View,
    StringEncoding::Fsst,
    StringEncoding::OnPair,
    StringEncoding::Zstd,
];
const STRING_STRUCTURES: &[StringStructure] = &[
    StringStructure::Flat,
    StringStructure::Dict,
    StringStructure::Chunked,
];
const STRING_VALIDITIES: &[StringValidity] =
    &[StringValidity::NonNullable, StringValidity::Nullable];
const ARROW_STRING_LAYOUTS: &[ArrowStringLayout] =
    &[ArrowStringLayout::Offset, ArrowStringLayout::View];

const STRING_ROWS: usize = 100_000;
const STRING_CHUNKS: usize = 4;
const DICTIONARY_SIZE: usize = 2_048;

fn string_cases() -> Vec<StringCase> {
    iproduct!(
        STRING_ENCODINGS.iter().copied(),
        STRING_STRUCTURES.iter().copied(),
        STRING_VALIDITIES.iter().copied()
    )
    .map(|(encoding, structure, validity)| StringCase {
        encoding,
        structure,
        validity,
    })
    .collect()
}

fn string_export_cases() -> Vec<StringExportCase> {
    iproduct!(string_cases(), ARROW_STRING_LAYOUTS.iter().copied())
        .map(|(array, layout)| StringExportCase { array, layout })
        .collect()
}

fn structured_strings(len: usize, validity: StringValidity) -> VarBinViewArray {
    match validity {
        StringValidity::NonNullable => {
            let values = (0..len)
                .map(|index| format!("https://example.com/common/path/{index:06}/shared-suffix"))
                .collect::<Vec<_>>();
            VarBinViewArray::from_iter_str(values.iter().map(String::as_str))
        }
        StringValidity::Nullable => {
            let values = (0..len)
                .map(|index| {
                    (!index.is_multiple_of(3)).then(|| {
                        format!("https://example.com/common/path/{index:06}/shared-suffix")
                    })
                })
                .collect::<Vec<_>>();
            VarBinViewArray::from_iter(
                values.iter().map(|value| value.as_deref()),
                DType::Utf8(Nullability::Nullable),
            )
        }
    }
}

fn dictionary_codes() -> ArrayRef {
    PrimitiveArray::from_iter(
        (0..STRING_ROWS).map(|index| u16::try_from(index % DICTIONARY_SIZE).unwrap()),
    )
    .into_array()
}

fn offset(source: VarBinViewArray, ctx: &mut ExecutionCtx) -> ArrayRef {
    let source = source.into_array();
    let mut builder = VarBinBuilder::<i32>::with_capacity_in(
        source.dtype().clone(),
        source.len(),
        ctx.allocator(),
    );
    source.append_to_builder(&mut builder, ctx).unwrap();
    builder.finish_into_varbin().into_array()
}

fn fsst(source: ArrayRef, ctx: &mut ExecutionCtx) -> ArrayRef {
    let compressor = fsst_train_compressor(&source, ctx).unwrap();
    fsst_compress(&source, &compressor, ctx)
        .unwrap()
        .into_array()
}

fn encode_strings(
    source: VarBinViewArray,
    encoding: StringEncoding,
    ctx: &mut ExecutionCtx,
) -> ArrayRef {
    match encoding {
        StringEncoding::Offset => offset(source, ctx),
        StringEncoding::View => source.into_array(),
        StringEncoding::Fsst => fsst(source.into_array(), ctx),
        StringEncoding::OnPair => {
            onpair_compress(&source.into_array(), DEFAULT_CONFIG, ctx).unwrap()
        }
        StringEncoding::Zstd => Zstd::from_var_bin_view_without_dict(&source, 3, 8_192, ctx)
            .unwrap()
            .into_array(),
    }
}

fn dictionary_strings(
    encoding: StringEncoding,
    validity: StringValidity,
    ctx: &mut ExecutionCtx,
) -> ArrayRef {
    let values = encode_strings(structured_strings(DICTIONARY_SIZE, validity), encoding, ctx);
    DictArray::try_new(dictionary_codes(), values)
        .unwrap()
        .into_array()
}

fn chunked_strings(
    encoding: StringEncoding,
    validity: StringValidity,
    ctx: &mut ExecutionCtx,
) -> ArrayRef {
    let chunk_size = STRING_ROWS / STRING_CHUNKS;
    let chunks = (0..STRING_CHUNKS).map(|chunk_index| {
        let start = chunk_index * chunk_size;
        let end = if chunk_index + 1 == STRING_CHUNKS {
            STRING_ROWS
        } else {
            start + chunk_size
        };
        encode_strings(structured_strings(end - start, validity), encoding, ctx)
    });
    ChunkedArray::try_new(chunks, DType::Utf8(validity.nullability()))
        .unwrap()
        .into_array()
}

fn string_array(case: StringCase) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    match case.structure {
        StringStructure::Flat => encode_strings(
            structured_strings(STRING_ROWS, case.validity),
            case.encoding,
            &mut ctx,
        ),
        StringStructure::Dict => dictionary_strings(case.encoding, case.validity, &mut ctx),
        StringStructure::Chunked => chunked_strings(case.encoding, case.validity, &mut ctx),
    }
}

/// Measures export to Arrow offset arrays and Arrow view arrays.
#[divan::bench(args = string_export_cases())]
fn string_export(bencher: Bencher, case: StringExportCase) {
    let array = string_array(case.array);
    let field = Field::new(
        "value",
        case.layout.data_type(),
        array.dtype().is_nullable(),
    );
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .input_counter(|(array, _)| ItemsCount::new(array.len()))
        .bench_values(|(array, mut ctx)| {
            SESSION
                .arrow()
                .execute_arrow(array, Some(&field), &mut ctx)
                .unwrap()
        });
}

/// Measures a direct append to an offset builder.
#[divan::bench(args = string_cases())]
fn append_to_varbin_builder(bencher: Bencher, case: StringCase) {
    let array = string_array(case);

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .input_counter(|(array, _)| ItemsCount::new(array.len()))
        .bench_values(|(array, mut ctx)| {
            let mut builder = VarBinBuilder::<i32>::with_capacity_in(
                array.dtype().clone(),
                array.len(),
                ctx.allocator(),
            );
            array.append_to_builder(&mut builder, &mut ctx).unwrap();
            builder.finish_into_varbin()
        });
}

/// Measures a direct append to a view builder.
#[divan::bench(args = string_cases())]
fn append_to_view_builder(bencher: Bencher, case: StringCase) {
    let array = string_array(case);

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .input_counter(|(array, _)| ItemsCount::new(array.len()))
        .bench_values(|(array, mut ctx)| {
            let mut builder = VarBinViewBuilder::with_capacity_in(
                array.dtype().clone(),
                array.len(),
                ctx.allocator().clone(),
            );
            array.append_to_builder(&mut builder, &mut ctx).unwrap();
            builder.finish_into_varbinview()
        });
}

#[derive(Clone, Copy)]
enum DecimalExportCase {
    CanonicalI64P18,
    /// The values of `CanonicalI64P18` under a dtype whose canonical width is `i128`.
    CanonicalI128P38,
    MspI32P18,
    /// `MspI32P18` with every third row null and zeros in the null slots.
    MspI32P18Nullable,
    /// Every value fits an `i16` but needs precision 5, above the precision 4 that `i16` holds.
    GapMspI16P10,
    /// A `Decimal(9, 2)` child with `i32` values under `Decimal(18, 2)`.
    WidenI32P18,
    /// A `Decimal(18, 2)` child with `i64` values under `Decimal(38, 2)`.
    WidenI64P38,
}

impl DecimalExportCase {
    fn decimal_dtype(self) -> DecimalDType {
        match self {
            Self::CanonicalI64P18
            | Self::MspI32P18
            | Self::MspI32P18Nullable
            | Self::WidenI32P18 => DecimalDType::new(18, 2),
            Self::CanonicalI128P38 | Self::WidenI64P38 => DecimalDType::new(38, 2),
            Self::GapMspI16P10 => DecimalDType::new(10, 2),
        }
    }

    fn array(self) -> ArrayRef {
        let decimal_dtype = self.decimal_dtype();
        match self {
            Self::CanonicalI64P18 => DecimalArray::new(
                spread_values::<i64>(-MAX_P18, MAX_P18),
                decimal_dtype,
                Validity::NonNullable,
            )
            .into_array(),
            Self::CanonicalI128P38 => DecimalArray::new(
                spread_values::<i128>(-MAX_P18, MAX_P18),
                decimal_dtype,
                Validity::NonNullable,
            )
            .into_array(),
            Self::MspI32P18 => decimal_byte_parts(
                PrimitiveArray::new(
                    spread_values::<i32>(-999_999_999, 999_999_999),
                    Validity::NonNullable,
                ),
                decimal_dtype,
            ),
            Self::MspI32P18Nullable => {
                let validity = Validity::from_iter((0..DECIMAL_ROWS).map(|index| index % 3 != 0));
                let values: Buffer<i32> = spread_values::<i32>(-999_999_999, 999_999_999)
                    .iter()
                    .enumerate()
                    .map(|(index, &value)| if index % 3 == 0 { 0 } else { value })
                    .collect();
                decimal_byte_parts(PrimitiveArray::new(values, validity), decimal_dtype)
            }
            Self::GapMspI16P10 => decimal_byte_parts(
                PrimitiveArray::new(
                    spread_values::<i16>(10_000, i64::from(i16::MAX)),
                    Validity::NonNullable,
                ),
                decimal_dtype,
            ),
            Self::WidenI32P18 => widen(
                DecimalArray::new(
                    spread_values::<i32>(-999_999_999, 999_999_999),
                    DecimalDType::new(9, 2),
                    Validity::NonNullable,
                ),
                decimal_dtype,
            ),
            Self::WidenI64P38 => widen(
                DecimalArray::new(
                    spread_values::<i64>(-MAX_P18, MAX_P18),
                    DecimalDType::new(18, 2),
                    Validity::NonNullable,
                ),
                decimal_dtype,
            ),
        }
    }
}

impl Display for DecimalExportCase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::CanonicalI64P18 => "canonical_i64_p18",
            Self::CanonicalI128P38 => "canonical_i128_p38",
            Self::MspI32P18 => "msp_i32_p18",
            Self::MspI32P18Nullable => "msp_i32_p18_nullable",
            Self::GapMspI16P10 => "gap_msp_i16_p10",
            Self::WidenI32P18 => "widen_i32_p18",
            Self::WidenI64P38 => "widen_i64_p38",
        })
    }
}

const DECIMAL_EXPORT_CASES: &[DecimalExportCase] = &[
    DecimalExportCase::CanonicalI64P18,
    DecimalExportCase::CanonicalI128P38,
    DecimalExportCase::MspI32P18,
    DecimalExportCase::MspI32P18Nullable,
    DecimalExportCase::GapMspI16P10,
    DecimalExportCase::WidenI32P18,
    DecimalExportCase::WidenI64P38,
];

const DECIMAL_ROWS: usize = 100_000;

/// The largest magnitude of a `Decimal(18, _)` value.
const MAX_P18: i64 = 999_999_999_999_999_999;

/// `DECIMAL_ROWS` values spread across `min..=max` in an unsorted, fixed order.
///
/// Row `i` holds `min + (i * STEP) % (max - min + 1)`. `STEP` is the prime `2^61 - 1`, so it is
/// coprime to every range size below it: consecutive rows land far apart, and the values repeat
/// only after every value of the range appears.
fn spread_values<T: TryFrom<i64>>(min: i64, max: i64) -> Buffer<T> {
    const STEP: i128 = (1 << 61) - 1;
    let count = i128::from(max) - i128::from(min) + 1;
    (0..DECIMAL_ROWS)
        .map(|index| {
            let offset = (i128::try_from(index).unwrap() * STEP) % count;
            let value = i64::try_from(i128::from(min) + offset).unwrap();
            T::try_from(value).ok().unwrap()
        })
        .collect()
}

fn decimal_byte_parts(msp: PrimitiveArray, decimal_dtype: DecimalDType) -> ArrayRef {
    DecimalByteParts::try_new(msp.into_array(), decimal_dtype)
        .unwrap()
        .into_array()
}

fn widen(values: DecimalArray, decimal_dtype: DecimalDType) -> ArrayRef {
    let dtype = DType::Decimal(decimal_dtype, values.dtype().nullability());
    WidenArray::try_new(values.into_array(), dtype)
        .unwrap()
        .into_array()
}

/// Measures export of one decimal array to an Arrow `Decimal128` array.
#[divan::bench(args = DECIMAL_EXPORT_CASES)]
fn decimal_export(bencher: Bencher, case: DecimalExportCase) {
    let array = case.array();
    let decimal_dtype = case.decimal_dtype();
    let field = Field::new(
        "value",
        DataType::Decimal128(decimal_dtype.precision(), decimal_dtype.scale()),
        array.dtype().is_nullable(),
    );
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .input_counter(|(array, _)| ItemsCount::new(array.len()))
        .bench_values(|(array, mut ctx)| {
            SESSION
                .arrow()
                .execute_arrow(array, Some(&field), &mut ctx)
                .unwrap()
        });
}

#[divan::bench]
fn to_arrow_array(bencher: Bencher) {
    bencher
        .with_inputs(|| (array(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            #[expect(deprecated, reason = "benchmarking deprecated code path")]
            array.execute_arrow(None, &mut ctx).unwrap()
        });
}

#[allow(non_snake_case)]
#[divan::bench]
fn ArrowExportVTable_execute_arrow(bencher: Bencher) {
    bencher
        .with_inputs(|| (array(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            SESSION
                .arrow()
                .execute_arrow(array, None, &mut ctx)
                .unwrap()
        })
}

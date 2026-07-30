// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_schema::DataType;
use arrow_schema::Field;
use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builders::VarBinBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::session::ArraySessionExt;
#[expect(
    deprecated,
    reason = "benchmark comparing deprecated method with new one"
)]
use vortex_arrow::ArrowArrayExecutor;
use vortex_arrow::ArrowSessionExt;
#[allow(deprecated)]
use vortex_arrow::dtype::ToArrowType as _;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_zstd::Zstd;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = array_session();
    vortex_fsst::initialize(&session);
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

#[divan::bench]
fn to_arrow_dtype(bencher: Bencher) {
    bencher.with_inputs(schema).bench_values(|dtype| {
        #[expect(deprecated, reason = "benchmarking deprecated code path")]
        dtype.to_arrow_dtype().unwrap()
    });
}

#[allow(non_snake_case)]
#[divan::bench]
fn ArrowExportVTable_to_arrow_field(bencher: Bencher) {
    bencher
        .with_inputs(schema)
        .bench_values(|dtype| SESSION.arrow().to_arrow_field("", &dtype).unwrap())
}

#[derive(Clone, Copy, Debug)]
enum StringEncoding {
    View,
    Fsst,
    Zstd,
    Dict,
    DictFsst,
    DictZstd,
    FilterFsst,
    FilterZstd,
    FilterDictFsst,
    ChunkedFsst,
    /// Every third row null, so the export walks a partial validity mask rather than an all-valid
    /// one and has to interleave nulls with the decoded values.
    NullableFsst,
    NullableZstd,
    NullableDict,
}

const STRING_ENCODINGS: &[StringEncoding] = &[
    StringEncoding::View,
    StringEncoding::Fsst,
    StringEncoding::Zstd,
    StringEncoding::Dict,
    StringEncoding::DictFsst,
    StringEncoding::DictZstd,
    StringEncoding::FilterFsst,
    StringEncoding::FilterZstd,
    StringEncoding::FilterDictFsst,
    StringEncoding::ChunkedFsst,
    StringEncoding::NullableFsst,
    StringEncoding::NullableZstd,
    StringEncoding::NullableDict,
];

/// Encodings whose `append_to_builder` the builder benchmarks reach directly.
///
/// The Arrow export cannot stand in for these: `execute_until` stops at the first canonical array,
/// so a bare FSST/Zstd root is canonicalized to `VarBinView` before any builder sees it.
/// Only `Chunked`, `Constant` and `VarBin` roots reach an encoding's own `append_to_builder` that
/// way, whereas the scan machinery appends encoded arrays into a builder directly.
const BUILDER_STRING_ENCODINGS: &[StringEncoding] = &[
    StringEncoding::View,
    StringEncoding::Fsst,
    StringEncoding::Zstd,
    StringEncoding::Dict,
    StringEncoding::ChunkedFsst,
    StringEncoding::NullableFsst,
    StringEncoding::NullableZstd,
    StringEncoding::NullableDict,
];

const OFFSET_STRING_ROWS: usize = 100_000;
const OFFSET_STRING_CHUNKS: usize = 4;
const DICTIONARY_SIZE: usize = 2_048;

fn structured_strings(len: usize) -> VarBinViewArray {
    let values = (0..len)
        .map(|index| format!("https://example.com/common/path/{index:06}/shared-suffix"))
        .collect::<Vec<_>>();
    VarBinViewArray::from_iter_str(values.iter().map(String::as_str))
}

fn nullable_structured_strings(len: usize) -> VarBinViewArray {
    let values = (0..len)
        .map(|index| {
            (!index.is_multiple_of(3))
                .then(|| format!("https://example.com/common/path/{index:06}/shared-suffix"))
        })
        .collect::<Vec<_>>();
    VarBinViewArray::from_iter(
        values.iter().map(|value| value.as_deref()),
        DType::Utf8(Nullability::Nullable),
    )
}

fn dictionary_values() -> VarBinViewArray {
    structured_strings(DICTIONARY_SIZE)
}

fn dictionary_codes() -> ArrayRef {
    PrimitiveArray::from_iter(
        (0..OFFSET_STRING_ROWS).map(|index| u16::try_from(index % DICTIONARY_SIZE).unwrap()),
    )
    .into_array()
}

fn half_rows_mask() -> Mask {
    Mask::from_iter((0..OFFSET_STRING_ROWS).map(|index| index.is_multiple_of(2)))
}

fn filtered(array: ArrayRef) -> ArrayRef {
    // Keep Filter as a lazy intermediate so benchmark setup cannot optimize it away.
    FilterArray::new(array, half_rows_mask()).into_array()
}

fn chunked_fsst(ctx: &mut ExecutionCtx) -> ArrayRef {
    let source = structured_strings(OFFSET_STRING_ROWS).into_array();
    let compressor = fsst_train_compressor(&source, ctx).unwrap();
    let chunk_size = OFFSET_STRING_ROWS / OFFSET_STRING_CHUNKS;
    let chunks = (0..OFFSET_STRING_CHUNKS).map(|chunk_index| {
        let start = chunk_index * chunk_size;
        let end = if chunk_index + 1 == OFFSET_STRING_CHUNKS {
            OFFSET_STRING_ROWS
        } else {
            start + chunk_size
        };
        let chunk = source.slice(start..end).unwrap();
        fsst_compress(&chunk, &compressor, ctx)
            .unwrap()
            .into_array()
    });
    ChunkedArray::try_new(chunks, source.dtype().clone())
        .unwrap()
        .into_array()
}

fn fsst(source: ArrayRef, ctx: &mut ExecutionCtx) -> ArrayRef {
    let compressor = fsst_train_compressor(&source, ctx).unwrap();
    fsst_compress(&source, &compressor, ctx)
        .unwrap()
        .into_array()
}

fn string_array(encoding: StringEncoding) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    match encoding {
        StringEncoding::View => structured_strings(OFFSET_STRING_ROWS).into_array(),
        StringEncoding::Fsst => fsst(
            structured_strings(OFFSET_STRING_ROWS).into_array(),
            &mut ctx,
        ),
        StringEncoding::Zstd => {
            let source = structured_strings(OFFSET_STRING_ROWS);
            Zstd::from_var_bin_view_without_dict(&source, 3, 8_192, &mut ctx)
                .unwrap()
                .into_array()
        }
        StringEncoding::Dict => {
            DictArray::try_new(dictionary_codes(), dictionary_values().into_array())
                .unwrap()
                .into_array()
        }
        StringEncoding::DictFsst => {
            let values = fsst(dictionary_values().into_array(), &mut ctx);
            DictArray::try_new(dictionary_codes(), values)
                .unwrap()
                .into_array()
        }
        StringEncoding::DictZstd => {
            let values = dictionary_values();
            let compressed_values =
                Zstd::from_var_bin_view_without_dict(&values, 3, 8_192, &mut ctx)
                    .unwrap()
                    .into_array();
            DictArray::try_new(dictionary_codes(), compressed_values)
                .unwrap()
                .into_array()
        }
        StringEncoding::FilterFsst => filtered(string_array(StringEncoding::Fsst)),
        StringEncoding::FilterZstd => filtered(string_array(StringEncoding::Zstd)),
        StringEncoding::FilterDictFsst => filtered(string_array(StringEncoding::DictFsst)),
        StringEncoding::ChunkedFsst => chunked_fsst(&mut ctx),
        StringEncoding::NullableFsst => fsst(
            nullable_structured_strings(OFFSET_STRING_ROWS).into_array(),
            &mut ctx,
        ),
        StringEncoding::NullableZstd => {
            let source = nullable_structured_strings(OFFSET_STRING_ROWS);
            Zstd::from_var_bin_view_without_dict(&source, 3, 8_192, &mut ctx)
                .unwrap()
                .into_array()
        }
        StringEncoding::NullableDict => {
            // Nulls live in the dictionary rather than the codes, so the export has to combine
            // the two validities.
            let values = nullable_structured_strings(DICTIONARY_SIZE);
            DictArray::try_new(dictionary_codes(), values.into_array())
                .unwrap()
                .into_array()
        }
    }
}

/// End-to-end export to Arrow `Utf8`, which is served through a `VarBinBuilder`.
#[divan::bench(args = STRING_ENCODINGS)]
fn offset_string_export(bencher: Bencher, encoding: StringEncoding) {
    let array = string_array(encoding);
    let field = Field::new("value", DataType::Utf8, array.dtype().is_nullable());

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

/// Appends an encoded array straight into an offset builder.
#[divan::bench(args = BUILDER_STRING_ENCODINGS)]
fn append_to_varbin_builder(bencher: Bencher, encoding: StringEncoding) {
    let array = string_array(encoding);

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .input_counter(|(array, _)| ItemsCount::new(array.len()))
        .bench_values(|(array, mut ctx)| {
            let mut builder =
                VarBinBuilder::<i32>::with_capacity(array.dtype().clone(), array.len());
            array.append_to_builder(&mut builder, &mut ctx).unwrap();
            builder.finish_into_varbin()
        });
}

/// Appends an encoded array straight into a view builder.
#[divan::bench(args = BUILDER_STRING_ENCODINGS)]
fn append_to_view_builder(bencher: Bencher, encoding: StringEncoding) {
    let array = string_array(encoding);

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .input_counter(|(array, _)| ItemsCount::new(array.len()))
        .bench_values(|(array, mut ctx)| {
            let mut builder = VarBinViewBuilder::with_capacity(array.dtype().clone(), array.len());
            array.append_to_builder(&mut builder, &mut ctx).unwrap();
            builder.finish_into_varbinview()
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

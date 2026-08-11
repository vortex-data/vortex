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
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builders::VarBinBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::scalar::Scalar;
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
use vortex_onpair::DEFAULT_CONFIG;
use vortex_onpair::onpair_compress;
use vortex_session::VortexSession;
use vortex_zstd::Zstd;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = array_session();
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
enum StringOperator {
    Identity,
    Filter,
    Take,
    Slice,
    Mask,
    Zip,
}

impl Display for StringOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Identity => "identity",
            Self::Filter => "filter",
            Self::Take => "take",
            Self::Slice => "slice",
            Self::Mask => "mask",
            Self::Zip => "zip",
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
    operator: StringOperator,
    validity: StringValidity,
}

impl Display for StringCase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{}/{}/{}",
            self.encoding, self.structure, self.operator, self.validity
        )
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
const STRING_OPERATORS: &[StringOperator] = &[
    StringOperator::Identity,
    StringOperator::Filter,
    StringOperator::Take,
    StringOperator::Slice,
    StringOperator::Mask,
    StringOperator::Zip,
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
        STRING_OPERATORS.iter().copied(),
        STRING_VALIDITIES.iter().copied()
    )
    .map(|(encoding, structure, operator, validity)| StringCase {
        encoding,
        structure,
        operator,
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

fn filtered(array: ArrayRef) -> ArrayRef {
    let mask = Mask::from_iter((0..array.len()).map(|index| index.is_multiple_of(2)));
    // Create a lazy FilterArray. The benchmark executes Filter during export.
    FilterArray::new(array, mask).into_array()
}

fn offset(source: VarBinViewArray, ctx: &mut ExecutionCtx) -> ArrayRef {
    let source = source.into_array();
    let mut builder = VarBinBuilder::<i32>::with_capacity(source.dtype().clone(), source.len());
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

fn take(array: ArrayRef) -> ArrayRef {
    let indices = PrimitiveArray::from_iter(
        (0..array.len())
            .step_by(2)
            .map(|index| u64::try_from(index).unwrap()),
    );
    // Create a lazy DictArray. The benchmark executes Take during export.
    DictArray::try_new(indices.into_array(), array)
        .unwrap()
        .into_array()
}

fn sliced(array: ArrayRef) -> ArrayRef {
    let start = array.len() / 4;
    let end = array.len() * 3 / 4;
    // Create a lazy SliceArray. The benchmark executes Slice during export.
    SliceArray::new(array, start..end).into_array()
}

fn mask_array(len: usize) -> ArrayRef {
    BoolArray::from_iter((0..len).map(|index| !index.is_multiple_of(3))).into_array()
}

fn masked(array: ArrayRef) -> ArrayRef {
    let mask = mask_array(array.len());
    array.mask(mask).unwrap()
}

fn zipped(array: ArrayRef) -> ArrayRef {
    let replacement = ConstantArray::new(
        Scalar::utf8("replacement", array.dtype().nullability()),
        array.len(),
    )
    .into_array();
    mask_array(array.len()).zip(array, replacement).unwrap()
}

fn apply_operator(array: ArrayRef, operator: StringOperator) -> ArrayRef {
    match operator {
        StringOperator::Identity => array,
        StringOperator::Filter => filtered(array),
        StringOperator::Take => take(array),
        StringOperator::Slice => sliced(array),
        StringOperator::Mask => masked(array),
        StringOperator::Zip => zipped(array),
    }
}

fn string_array(case: StringCase) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let array = match case.structure {
        StringStructure::Flat => encode_strings(
            structured_strings(STRING_ROWS, case.validity),
            case.encoding,
            &mut ctx,
        ),
        StringStructure::Dict => dictionary_strings(case.encoding, case.validity, &mut ctx),
        StringStructure::Chunked => chunked_strings(case.encoding, case.validity, &mut ctx),
    };
    apply_operator(array, case.operator)
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
            let mut builder =
                VarBinBuilder::<i32>::with_capacity(array.dtype().clone(), array.len());
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

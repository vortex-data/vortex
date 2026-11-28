// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StringArray;
use divan::Bencher;
use mimalloc::MiMalloc;
use parquet_variant::Variant;
use parquet_variant::VariantBuilderExt;
use parquet_variant_compute::VariantArrayBuilder;
use parquet_variant_compute::json_to_variant;
use vortex_array::ArrayRef;
use vortex_array::Executable;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_arrow::ArrowSessionExt;
use vortex_functions::variant_get::VariantGet;
use vortex_functions::variant_get::VariantGetOptions;
use vortex_functions::variant_get::VariantPath;
use vortex_functions::variant_get::VariantPathElement;
use vortex_parquet_variant::ParquetVariant;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const ROWS: usize = 4096;
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_parquet_variant::initialize(&session);
    session
});
static INPUT: LazyLock<ArrayRef> = LazyLock::new(|| {
    let json: ArrowArrayRef = Arc::new(StringArray::from_iter_values((0..ROWS).map(|i| {
        match i % 16 {
            0 => r#"{"metric":{}}"#.to_owned(),
            1 => r#"{"metric":{"value":null}}"#.to_owned(),
            2 => format!(r#"{{"metric":{{"value":{i}.25}}}}"#),
            _ => format!(r#"{{"metric":{{"value":{i}}}}}"#),
        }
    })));
    ParquetVariant::from_arrow_variant(&json_to_variant(&json).unwrap(), &SESSION.arrow()).unwrap()
});

static BINARY_BOOL_INPUT: LazyLock<ArrayRef> = LazyLock::new(|| {
    let rows = if cfg!(codspeed) { 256 } else { 4096 };
    let mut builder = VariantArrayBuilder::new(rows);
    let payload = [0xff; 40];
    for i in 0..rows {
        let mut object = builder.new_object();
        let mut metric = object.new_object("metric");
        match i % 16 {
            0 => {}
            1 => {
                metric.insert("bytes", Variant::Null);
                metric.insert("flag", Variant::Null);
            }
            _ => {
                let bytes = match i % 16 {
                    2 => Variant::from("text"),
                    _ if i % 2 == 0 => Variant::Binary(&payload[..5]),
                    _ => Variant::Binary(&payload),
                };
                let flag = match i % 16 {
                    2 => Variant::from("true"),
                    3 => Variant::from("other"),
                    4 => Variant::Int8(-1),
                    _ => Variant::from(i % 2 == 0),
                };
                metric.insert("bytes", bytes);
                metric.insert("flag", flag);
            }
        }
        metric.finish();
        object.finish();
    }
    ParquetVariant::from_arrow_variant(&builder.build(), &SESSION.arrow()).unwrap()
});

fn main() {
    divan::main();
}

#[divan::bench(args = [PType::I32, PType::I64, PType::U64, PType::F64])]
fn numeric_field(bencher: Bencher, ptype: PType) {
    bench_field::<PrimitiveArray>(
        bencher,
        &INPUT,
        "value",
        DType::Primitive(ptype, Nullability::Nullable),
    );
}

#[divan::bench]
fn binary_field(bencher: Bencher) {
    bench_field::<VarBinViewArray>(
        bencher,
        &BINARY_BOOL_INPUT,
        "bytes",
        DType::Binary(Nullability::Nullable),
    );
}

#[divan::bench]
fn boolean_field(bencher: Bencher) {
    bench_field::<BoolArray>(
        bencher,
        &BINARY_BOOL_INPUT,
        "flag",
        DType::Bool(Nullability::Nullable),
    );
}

fn bench_field<E: Executable>(bencher: Bencher, input: &ArrayRef, field: &str, dtype: DType) {
    let path = VariantPath::new([
        VariantPathElement::field("metric"),
        VariantPathElement::field(field),
    ]);
    let expr = VariantGet::try_new(input.clone(), VariantGetOptions::new(path, Some(dtype)))
        .unwrap()
        .into_array();

    bencher
        .counter(divan::counter::ItemsCount::new(input.len()))
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| expr.clone().execute::<E>(ctx).unwrap());
}

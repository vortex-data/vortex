// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StringArray;
use divan::Bencher;
use mimalloc::MiMalloc;
use parquet_variant_compute::json_to_variant;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
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

fn main() {
    divan::main();
}

#[divan::bench(args = [PType::I32, PType::I64, PType::U64, PType::F64])]
fn numeric_field(bencher: Bencher, ptype: PType) {
    let path = VariantPath::new([
        VariantPathElement::field("metric"),
        VariantPathElement::field("value"),
    ]);
    let dtype = DType::Primitive(ptype, Nullability::Nullable);
    let expr = VariantGet::try_new(INPUT.clone(), VariantGetOptions::new(path, Some(dtype)))
        .unwrap()
        .into_array();

    bencher
        .counter(divan::counter::ItemsCount::new(ROWS))
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| expr.clone().execute::<PrimitiveArray>(ctx).unwrap());
}

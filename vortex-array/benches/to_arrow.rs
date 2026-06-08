// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
#[expect(
    deprecated,
    reason = "benchmark comparing deprecated method with new one"
)]
use vortex_array::arrow::ArrowArrayExecutor;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::session::ArraySession;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

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
    // Warm the ArrowSession
    drop(SESSION.arrow().to_arrow_field("", &schema()).unwrap());

    bencher
        .with_inputs(schema)
        .bench_values(|dtype| SESSION.arrow().to_arrow_field("", &dtype).unwrap())
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
    // Warm the ArrowSession
    drop(
        SESSION
            .arrow()
            .execute_arrow(array(), None, &mut SESSION.create_execution_ctx()),
    );

    bencher
        .with_inputs(|| (array(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            SESSION
                .arrow()
                .execute_arrow(array, None, &mut ctx)
                .unwrap()
        })
}

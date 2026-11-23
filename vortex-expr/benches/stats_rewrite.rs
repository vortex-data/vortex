// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for stats-based predicate proving across a multi-file scan.
//!
//! A scan binds its filter once against the table schema and passes that bound expression to
//! every file reader, so each iteration measures proving a freshly bound filter across 64 files
//! (one cache miss followed by 63 session-cache hits).

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt_eq;
use vortex_array::expr::is_not_null;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::or;
use vortex_array::expr::root;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_expr::stats::session::StatsSession;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const NUM_FILES: usize = 64;
const COLUMN_COUNTS: &[usize] = &[8, 64, 512, 2048];

fn schema(num_columns: usize) -> DType {
    let names = FieldNames::from_iter((0..num_columns).map(|i| format!("c{i}")));
    let dtypes = vec![DType::Primitive(PType::I64, Nullability::Nullable); num_columns];
    DType::Struct(StructFields::new(names, dtypes), Nullability::NonNullable)
}

fn filter_expr() -> Expression {
    and(
        and(
            gt_eq(get_item("c0", root()), lit(10i64)),
            lt(get_item("c1", root()), lit(100i64)),
        ),
        or(
            eq(get_item("c2", root()), lit(42i64)),
            is_not_null(get_item("c3", root())),
        ),
    )
}

#[divan::bench(args = COLUMN_COUNTS)]
fn falsify(bencher: Bencher, num_columns: usize) {
    let dtype = schema(num_columns);
    let filter = filter_expr();
    let session = VortexSession::empty().with::<StatsSession>();

    bencher
        .with_inputs(|| filter.bind(&dtype).unwrap())
        .bench_local_refs(|expr| {
            for _ in 0..NUM_FILES {
                divan::black_box(expr.falsify(&session).unwrap());
            }
        });
}

#[divan::bench(args = COLUMN_COUNTS)]
fn satisfy(bencher: Bencher, num_columns: usize) {
    let dtype = schema(num_columns);
    let filter = filter_expr();
    let session = VortexSession::empty().with::<StatsSession>();

    bencher
        .with_inputs(|| filter.bind(&dtype).unwrap())
        .bench_local_refs(|expr| {
            for _ in 0..NUM_FILES {
                divan::black_box(expr.satisfy(&session).unwrap());
            }
        });
}

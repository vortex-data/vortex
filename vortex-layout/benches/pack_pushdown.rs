// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Times optimizing an expression over a wide struct, which pushes it through the struct's `Pack`.

#![expect(clippy::unwrap_used)]

use std::hint::black_box;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::checked_add;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::layouts::struct_::StructLayout;
use vortex_layout::plan::physical::EvalPlan;
use vortex_layout::plan::physical::PlanRef;
use vortex_layout::plan::physical::lower;
use vortex_layout::plan::physical::optimize;
use vortex_layout::segments::SegmentId;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<LayoutSession>());

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const COLUMNS: &[usize] = &[8, 105, 512];

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

fn struct_plan(columns: usize) -> PlanRef {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let fields =
        StructFields::from_iter((0..columns).map(|index| (format!("c{index}"), dtype.clone())));
    let children = (0..columns)
        .map(|index| {
            FlatLayout::new(
                1024,
                dtype.clone(),
                SegmentId::from(u32::try_from(index).unwrap()),
                ReadContext::new([]),
            )
            .into_layout()
        })
        .collect();
    let layout =
        StructLayout::new(1024, DType::Struct(fields, Nullability::NonNullable), children)
            .into_layout();
    lower(&layout).unwrap()
}

fn run(bencher: Bencher, columns: usize, expression: Expression) {
    let plan = struct_plan(columns);
    let expression = expression.bind(plan.dtype()).unwrap();
    bencher
        .with_inputs(|| {
            EvalPlan::try_new(expression.clone(), plan.clone())
                .unwrap()
                .into_plan()
        })
        .bench_values(|eval| black_box(optimize(eval, &SESSION).unwrap()));
}

/// A filter over one column.
#[divan::bench(args = COLUMNS)]
fn one_column_filter(bencher: Bencher, columns: usize) {
    run(bencher, columns, gt(get_item("c3", root()), lit(5_i32)));
}

/// A filter over two columns, which partitions into two field expressions.
#[divan::bench(args = COLUMNS)]
fn two_column_filter(bencher: Bencher, columns: usize) {
    run(
        bencher,
        columns,
        and(
            gt(get_item("c1", root()), lit(5_i32)),
            gt(get_item("c4", root()), lit(7_i32)),
        ),
    );
}

/// A projection of three columns, one computed.
#[divan::bench(args = COLUMNS)]
fn three_column_projection(bencher: Bencher, columns: usize) {
    run(
        bencher,
        columns,
        pack(
            [
                ("a", get_item("c0", root())),
                ("b", get_item("c2", root())),
                (
                    "c",
                    checked_add(get_item("c5", root()), get_item("c6", root())),
                ),
            ],
            Nullability::NonNullable,
        ),
    );
}

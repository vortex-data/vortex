// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::expr::list_contains;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar::Scalar;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

/// One engine-sized chunk against a handful of representative literal `IN`-list sizes, all at or
/// past `MIN_ELEMENTS_FOR_SORTED_MEMBERSHIP` so `sorted_merge` genuinely takes the fast path
/// (below that threshold both benches run the identical fan-out, which isn't an interesting
/// comparison -- that crossover was measured separately to pick the threshold, see
/// `MIN_ELEMENTS_FOR_SORTED_MEMBERSHIP`'s doc comment). The fan-out (`col IN (a, b, c, ...)` as
/// one `Eq` pass per literal, `Or`-reduced) is `O(list_len * ROWS)`; the sorted merge is
/// `O(ROWS log list_len)` plus a one-time `O(list_len log list_len)` sort of the literal set.
/// Both are benchmarked against the same sorted column, gated only by whether `Stat::IsSorted`
/// has been computed on it, matching how `ListContains::execute` actually chooses between them.
const ROWS: usize = 8_192;
const LIST_LENS: &[usize] = &[16, 64, 256];

fn list_scalar(list_len: usize) -> Scalar {
    Scalar::list(
        std::sync::Arc::new(vortex_array::dtype::DType::Primitive(
            vortex_array::dtype::PType::I64,
            vortex_array::dtype::Nullability::NonNullable,
        )),
        // Sparse, non-adjacent literals spread across the probed range so every comparison
        // in the fan-out genuinely has to run (no early all-true/all-false short circuit).
        (0..list_len)
            .map(|i| Scalar::from((i * (ROWS / list_len.max(1))) as i64))
            .collect(),
        vortex_array::dtype::Nullability::NonNullable,
    )
}

fn sorted_column() -> ArrayRef {
    let arr = PrimitiveArray::from_iter(0_i64..ROWS as i64).into_array();
    arr.statistics()
        .compute_is_sorted(&mut SESSION.create_execution_ctx());
    arr
}

#[divan::bench(args = LIST_LENS)]
fn fanout(bencher: Bencher, list_len: usize) {
    // A column that has never had `Stat::IsSorted` computed: `ListContains::execute` takes the
    // pre-existing equality fan-out.
    let column = PrimitiveArray::from_iter(0_i64..ROWS as i64).into_array();
    let expr = list_contains(lit(list_scalar(list_len)), root());
    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| (column.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(column, ctx)| {
            column
                .clone()
                .apply(&expr)
                .unwrap()
                .execute::<BoolArray>(ctx)
                .unwrap()
        });
}

#[divan::bench(args = LIST_LENS)]
fn sorted_merge(bencher: Bencher, list_len: usize) {
    let column = sorted_column();
    let expr = list_contains(lit(list_scalar(list_len)), root());
    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| (column.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(column, ctx)| {
            column
                .clone()
                .apply(&expr)
                .unwrap()
                .execute::<BoolArray>(ctx)
                .unwrap()
        });
}
